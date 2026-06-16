/**
 * @file m20_highlevel_policy_runner.hpp
 * @brief High-level residual policy runner for M20 (cascaded with low-level policy)
 * @author Bo (Percy) Peng
 * @version 1.0
 * @date 2025-11-07
 *
 * @copyright Copyright (c) 2025  DeepRobotics
 *
 */

#pragma once
#define PI 3.14159265358979323846

#include "policy_runner_base.hpp"
#include <ctime>
#include <cmath>
#include <utility>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

class M20HighLevelPolicyRunner : public PolicyRunnerBase {
private:
    VecXf kp_, kd_;
    VecXf dof_default_eigen_policy, dof_default_eigen_robot;
    Vec3f gravity_direction = Vec3f(0., 0., -1.);
    timespec system_time;

    const int motor_num = 16;
    const int leg_num = 12;
    const int ll_one_step_obs_dim = 57;  // 底层单步obs: cmd(3)+ang_vel(3)+gravity(3)+joint_pos(16)+joint_vel(16)+actions(16)
    const int hl_one_step_obs_dim = 73;  // 高层单步obs: 底层obs(57) + ll_actions(16)
    const int history_length = 5;
    const int ll_observation_dim = ll_one_step_obs_dim * history_length; // 285
    const int hl_observation_dim = hl_one_step_obs_dim * history_length; // 365
    const int action_dim = 16;
    float agent_timestep = 0.02;

    VecXf joint_pos_rl = VecXf(action_dim);
    VecXf joint_vel_rl = VecXf(action_dim);

    const std::string ll_policy_path_;
    const std::string hl_policy_path_;

    float omega_scale_ = 0.25;
    float dof_vel_scale_ = 0.05;

    // 底层 obs 历史缓冲区
    // 训练中 flatten_history_dim=False，compute_group 返回 (N, 5, 57)，
    // 然后 flip(1).reshape(N, -1) 得到展平布局:
    //   [all_terms(t), all_terms(t-1), ..., all_terms(t-4)]
    // 即每个时刻的所有 term 连续存放，历史帧从新到旧排列
    VecXf ll_one_step_obs_;         // 底层单步 obs (57维)
    VecXf ll_obs_history_;          // 底层历史 obs 缓冲区 (285维)
    VecXf ll_last_action_eigen;     // 底层上一步动作 (16维)
    VecXf ll_current_action_;       // 底层当前输出 (16维)

    VecXf hl_one_step_obs_;       // 高层单步观测 (73维)
    VecXf hl_obs_history_;        // 高层历史观测缓冲区 (365维)
    VecXf hl_last_action_eigen;   // 高层上一步动作 (16维，高层自身的输出)
    VecXf hl_current_action_;     // 高层当前输出 (16维)

    VecXf tmp_action_eigen;       // 最终动作（robot order）
    VecXf final_action_policy;    // 最终动作（policy order）

    RobotAction robot_action;
    std::vector<std::string> robot_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint", "fl_wheel_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint", "fr_wheel_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint", "hl_wheel_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint", "hr_wheel_joint"};

    std::vector<std::string> policy_order = {
        "fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint",
        "fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint",
        "hl_hipx_joint", "hl_hipy_joint", "hl_knee_joint",
        "hr_hipx_joint", "hr_hipy_joint", "hr_knee_joint",
        "fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint",
    };

    std::vector<float> action_scale_robot = {0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5,
                                             0.125, 0.25, 0.25, 5};

    // 底层 ONNX session
    Ort::Env ll_env_;
    Ort::SessionOptions ll_session_options_;
    Ort::Session ll_session_{nullptr};

    // 高层 ONNX session
    Ort::Env hl_env_;
    Ort::SessionOptions hl_session_options_;
    Ort::Session hl_session_{nullptr};

    std::vector<int> robot2policy_idx, policy2robot_idx;

    const char* input_names_[1] = {"obs"};
    const char* output_names_[1] = {"actions"};

    Ort::MemoryInfo memory_info{nullptr};
    const std::array<int64_t, 2> ll_input_shape = {1, ll_observation_dim};
    const std::array<int64_t, 2> hl_input_shape = {1, hl_observation_dim};

    float hl_clip_actions_ = 5.0;  // 与训练时 clip_actions 一致
    float time_step = 0.;
    bool first_frame_ = true;      // CircularBuffer 首次 append 填充标志

public:
    M20HighLevelPolicyRunner(const std::string &policy_name,
                             const std::string &ll_policy_path,
                             const std::string &hl_policy_path) :
            PolicyRunnerBase(policy_name),
            ll_policy_path_(ll_policy_path),
            hl_policy_path_(hl_policy_path),
            ll_env_(ORT_LOGGING_LEVEL_WARNING, "M20LLPolicy"),
            hl_env_(ORT_LOGGING_LEVEL_WARNING, "M20HLPolicy"),
            ll_session_options_{},
            hl_session_options_{},
            ll_session_{nullptr},
            hl_session_{nullptr},
            memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {

        dof_default_eigen_policy.setZero(action_dim);
        dof_default_eigen_robot.setZero(action_dim);
        dof_default_eigen_policy << 0.0, -0.6,  1.0,
                                    0.0, -0.6,  1.0,
                                    0.0,  0.6, -1.0,
                                    0.0,  0.6, -1.0,
                                    0.0, 0.0, 0.0, 0.0;
        dof_default_eigen_robot << 0.0, -0.6,  1.0, 0.0,
                                   0.0, -0.6,  1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0,
                                   0.0,  0.6, -1.0, 0.0;
        SetDecimation(4);

        // 底层 session 配置
        ll_session_options_.SetIntraOpNumThreads(4);
        ll_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (access(ll_policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "Low-level model file not found: " << ll_policy_path_ << std::endl;
            throw std::runtime_error("Low-level model file missing");
        }
        ll_session_ = Ort::Session(ll_env_, ll_policy_path_.c_str(), ll_session_options_);

        // 高层 session 配置
        hl_session_options_.SetIntraOpNumThreads(4);
        hl_session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        if (access(hl_policy_path_.c_str(), F_OK) != 0) {
            std::cerr << "High-level model file not found: " << hl_policy_path_ << std::endl;
            throw std::runtime_error("High-level model file missing");
        }
        hl_session_ = Ort::Session(hl_env_, hl_policy_path_.c_str(), hl_session_options_);

        kp_ = Vec4f(80, 80, 80, 0.).replicate(4, 1);
        kd_ = Vec4f(2, 2, 2, 0.6).replicate(4, 1);

        robot2policy_idx = generate_permutation(robot_order, policy_order);
        policy2robot_idx = generate_permutation(policy_order, robot_order);

        robot_action.kp = kp_;
        robot_action.kd = kd_;
        robot_action.tau_ff = VecXf::Zero(motor_num);
        robot_action.goal_joint_pos = VecXf::Zero(motor_num);
        robot_action.goal_joint_vel = VecXf::Zero(motor_num);

        // 初始化缓冲区
        ll_one_step_obs_.setZero(ll_one_step_obs_dim);
        ll_obs_history_.setZero(ll_observation_dim);
        ll_last_action_eigen.setZero(action_dim);
        ll_current_action_.setZero(action_dim);

        hl_one_step_obs_.setZero(hl_one_step_obs_dim);
        hl_obs_history_.setZero(hl_observation_dim);
        hl_last_action_eigen.setZero(action_dim);
        hl_current_action_.setZero(action_dim);

        tmp_action_eigen.setZero(action_dim);
        final_action_policy.setZero(action_dim);

        memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        std::cout << "[M20HighLevelPolicyRunner] Loaded cascaded policy (LL: "
                  << ll_policy_path_ << ", HL: " << hl_policy_path_ << ")" << std::endl;
    }

    ~M20HighLevelPolicyRunner() override = default;

    std::vector<int> generate_permutation(
        const std::vector<std::string>& from,
        const std::vector<std::string>& to,
        int default_index = 0)
    {
        std::unordered_map<std::string, int> idx_map;
        for (int i = 0; i < (int)from.size(); ++i) {
            idx_map[from[i]] = i;
        }
        std::vector<int> perm;
        for (const auto& name : to) {
            auto it = idx_map.find(name);
            if (it != idx_map.end()) {
                perm.push_back(it->second);
            } else {
                perm.push_back(default_index);
            }
        }
        return perm;
    }

    void DisplayPolicyInfo() {}

    void OnEnter() {
        run_cnt_ = 0;
        cmd_vel_input_.setZero();
        first_frame_ = true;
        ll_one_step_obs_.setZero(ll_one_step_obs_dim);
        ll_obs_history_.setZero(ll_observation_dim);
        ll_last_action_eigen.setZero(action_dim);
        ll_current_action_.setZero(action_dim);
        hl_one_step_obs_.setZero(hl_one_step_obs_dim);
        hl_obs_history_.setZero(hl_observation_dim);
        hl_last_action_eigen.setZero(action_dim);
        hl_current_action_.setZero(action_dim);
        tmp_action_eigen.setZero(action_dim);
        final_action_policy.setZero(action_dim);
    }

    VecXf OnnxInferLL(VecXf observation) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            observation.data(),
            observation.size(),
            ll_input_shape.data(),
            ll_input_shape.size()
        );
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));
        auto outputs = ll_session_.Run(
            Ort::RunOptions{nullptr},
            input_names_, inputs.data(), 1,
            output_names_, 1
        );
        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);
    }

    VecXf OnnxInferHL(VecXf observation) {
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            observation.data(),
            observation.size(),
            hl_input_shape.data(),
            hl_input_shape.size()
        );
        std::vector<Ort::Value> inputs;
        inputs.emplace_back(std::move(input_tensor));
        auto outputs = hl_session_.Run(
            Ort::RunOptions{nullptr},
            input_names_, inputs.data(), 1,
            output_names_, 1
        );
        float* action_data = outputs[0].GetTensorMutableData<float>();
        Eigen::Map<Eigen::VectorXf> action_map(action_data, action_dim);
        return VecXf(action_map);
    }

    RobotAction getRobotAction(const RobotBasicState &ro, const UserCommand &uc) {

        Vec3f base_omega = ro.base_omega * omega_scale_;
        Vec3f projected_gravity = ro.base_rot_mat.inverse() * gravity_direction;
        Vec3f command = Vec3f(uc.forward_vel_scale, uc.side_vel_scale, uc.turnning_vel_scale);

        for (int i = 0; i < action_dim; ++i) {
            joint_pos_rl(i) = ro.joint_pos(robot2policy_idx[i]);
            joint_vel_rl(i) = ro.joint_vel(robot2policy_idx[i]) * dof_vel_scale_;
        }
        joint_pos_rl.segment(12, 4).setZero();
        joint_pos_rl -= dof_default_eigen_policy;

        // === 步骤1: 构建底层 obs 并推理 ===
        // 训练中 flatten_history_dim=False 时 compute_group 返回 (N, 5, 57)，
        // 然后 flip(1).reshape(N, -1) 得到展平布局:
        //   [all_terms(t), all_terms(t-1), ..., all_terms(t-4)]
        // 即每个时刻的所有 term 连续存放，历史帧从新到旧排列
        // 这与 flatten_history_dim=True (per-term 历史拼接) 的布局完全不同!

        // 底层 obs (57维): cmd + ang_vel + gravity + joint_pos + joint_vel + ll_last_action
        // 历史缓冲区布局 (与训练时 CTSVecEnvWrapper .flip([1]).reshape() 一致):
        //   [all_terms(t), all_terms(t-1), ..., all_terms(t-4)]  newest first
        // 训练时 CircularBuffer 首次 append 用第一帧填充所有槽位:
        //   step 0: [obs_0, obs_0, obs_0, obs_0, obs_0]
        //   step 1: [obs_1, obs_0, obs_0, obs_0, obs_0]
        ll_one_step_obs_ << command,
                            base_omega,
                            projected_gravity,
                            joint_pos_rl,
                            joint_vel_rl,
                            ll_last_action_eigen;

        // Isaac Lab CircularBuffer 首次 append 行为：用第一帧填充所有历史槽位
        //   step 0: [obs_0, obs_0, obs_0, obs_0, obs_0]
        //   step 1: [obs_1, obs_0, obs_0, obs_0, obs_0]  (正常 shift+prepend)
        if (first_frame_) {
            for (int i = 0; i < history_length; ++i) {
                ll_obs_history_.segment(i * ll_one_step_obs_dim, ll_one_step_obs_dim) = ll_one_step_obs_;
            }
        } else {
            // 正常 shift right + prepend (newest first)
            ll_obs_history_.tail(ll_observation_dim - ll_one_step_obs_dim) =
                ll_obs_history_.head(ll_observation_dim - ll_one_step_obs_dim).eval();
            ll_obs_history_.head(ll_one_step_obs_dim) = ll_one_step_obs_;
        }

        // 底层推理
        ll_current_action_ = OnnxInferLL(ll_obs_history_);
        ll_last_action_eigen = ll_current_action_;

        // === 步骤2: 构建高层 obs 并推理 ===
        // 高层 obs: cmd + ang_vel + gravity + joint_pos + joint_vel + hl_last_action(高层自身上一步输出) + ll_actions(底层当前输出)
        // 训练时 Isaac Lab step() 内部执行顺序：
        //   1. action_manager.process_action(actions)
        //   2. scene.update(dt)
        //   3. observation_manager.compute() → last_ll_action() 触发 LL 推理，ll_actions 返回当前步 LL 输出
        // 因此 HL obs 中的 ll_actions 是当前步的 LL 输出（与 C++ 顺序推理一致）
        hl_one_step_obs_ << command,
                            base_omega,
                            projected_gravity,
                            joint_pos_rl,
                            joint_vel_rl,
                            hl_last_action_eigen,
                            ll_current_action_;

        // 高层历史缓冲区更新 (与 LL 同样的 CircularBuffer 首帧填充逻辑)
        // 注意: HL 的首帧填充在 LL 推理之后执行，因此 ll_current_action_ 已是当前步的正确值
        if (first_frame_) {
            for (int i = 0; i < history_length; ++i) {
                hl_obs_history_.segment(i * hl_one_step_obs_dim, hl_one_step_obs_dim) = hl_one_step_obs_;
            }
            first_frame_ = false;
        } else {
            hl_obs_history_.tail(hl_observation_dim - hl_one_step_obs_dim) =
                hl_obs_history_.head(hl_observation_dim - hl_one_step_obs_dim).eval();
            hl_obs_history_.head(hl_one_step_obs_dim) = hl_one_step_obs_;
        }

        // 高层推理
        hl_current_action_ = OnnxInferHL(hl_obs_history_);
        // clip 高层输出（与训练时 clip_actions=5.0 一致）
        hl_current_action_ = hl_current_action_.cwiseMax(-hl_clip_actions_).cwiseMin(hl_clip_actions_);
        hl_last_action_eigen = hl_current_action_;

        // === 步骤3: 合并最终动作 ===
        // 训练代码中: total_actions = low_level_actions + raw_actions (residual)
        // 腿关节(前12维): 底层输出 + 高层残差
        // 轮子(后4维): 底层输出 + 高层残差
        final_action_policy.head(leg_num) = ll_current_action_.head(leg_num) + hl_current_action_.head(leg_num);
        final_action_policy.tail(4) = ll_current_action_.tail(4) + hl_current_action_.tail(4);

        // 从 policy order 映射到 robot order，并应用 action scale
        for (int i = 0; i < action_dim; ++i) {
            tmp_action_eigen(i) = final_action_policy(policy2robot_idx[i]);
            tmp_action_eigen(i) *= action_scale_robot[i];
        }
        tmp_action_eigen += dof_default_eigen_robot;

        // 设置关节目标
        for (int i = 0; i < 4; ++i) {
            robot_action.goal_joint_pos.segment(i * 4, 3) = tmp_action_eigen.segment(i * 4, 3);
            robot_action.goal_joint_vel(i * 4 + 3) = tmp_action_eigen(i * 4 + 3);
        }

        // Debug: 打印前10步的关键数值用于和 play 对比
        if (run_cnt_ < 10) {
            std::cout << "[HL_DEBUG] step=" << run_cnt_
                      << " cmd=" << command.transpose()
                      << " ll_out=" << ll_current_action_.head(4).transpose()
                      << " hl_out=" << hl_current_action_.head(4).transpose()
                      << " final_leg0-3=" << final_action_policy.head(4).transpose()
                      << std::endl;
        }

        ++run_cnt_;
        ++time_step;
        return robot_action;
    }

    double getCurrentTime() {
        clock_gettime(1, &system_time);
        return system_time.tv_sec + system_time.tv_nsec / 1e9;
    }
};
