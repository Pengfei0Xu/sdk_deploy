// xbox_gamepad_interface.hpp
#pragma once

#include "user_command_interface.h"
#include "custom_types.h"
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>
#include <libevdev-1.0/libevdev/libevdev.h>

using namespace interface;
using namespace types;

class XboxGamepadInterface : public UserCommandInterface
{
private:
    std::atomic<bool> running_{false};
    std::thread gp_thread_;

    float max_forward_ = 1.0f;//
    float max_side_    = 1.0f;
    float max_yaw_     = 1.0f;
    float deadzone_    = 0.1f;

    struct libevdev* dev_ = nullptr;
    int fd_ = -1;

    void ClipNumber(float& num, float low, float high)
    {
        if (num < low) num = low;
        if (num > high) num = high;
    }

    float NormalizeAxis(int value, const struct input_absinfo* info)
    {
        float mid = (info->maximum + info->minimum) / 2.0f;
        float range = (info->maximum - info->minimum) / 2.0f;
        if (range < 1.0f) return 0.0f;
        float normalized = (value - mid) / range;
        if (std::fabs(normalized) < deadzone_) return 0.0f;
        return normalized;
    }

    bool FindAndOpenDevice()
    {
        DIR* dir = opendir("/dev/input");
        if (!dir) {
            std::cerr << "[XboxGamepad] Cannot open /dev/input\n";
            return false;
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name(entry->d_name);
            if (name.find("event") != 0) continue;

            std::string path = "/dev/input/" + name;
            int fd = open(path.c_str(), O_RDONLY | O_NONBLOCK);
            if (fd < 0) continue;

            struct libevdev* dev = nullptr;
            int rc = libevdev_new_from_fd(fd, &dev);
            if (rc < 0) {
                close(fd);
                continue;
            }

            // Check if device has analog sticks (gamepad) and is not a keyboard/mouse
            bool has_abs_x = libevdev_has_event_type(dev, EV_ABS) &&
                             libevdev_has_event_code(dev, EV_ABS, ABS_X);
            bool has_keys = libevdev_has_event_code(dev, EV_KEY, BTN_SOUTH) ||
                            libevdev_has_event_code(dev, EV_KEY, BTN_A);

            if (has_abs_x && has_keys) {
                const char* dev_name = libevdev_get_name(dev);
                std::cout << "[XboxGamepad] Found gamepad: " << (dev_name ? dev_name : "unknown")
                          << " at " << path << "\n";
                dev_ = dev;
                fd_ = fd;
                closedir(dir);
                return true;
            }

            libevdev_free(dev);
            close(fd);
        }

        closedir(dir);
        std::cerr << "[XboxGamepad] No gamepad device found\n";
        return false;
    }

    void CloseDevice()
    {
        if (dev_) {
            libevdev_free(dev_);
            dev_ = nullptr;
        }
        if (fd_ >= 0) {
            close(fd_);
            fd_ = -1;
        }
    }

    void gamepad_loop()
    {
        if (!FindAndOpenDevice()) {
            std::cerr << "[XboxGamepad] Failed to find Xbox gamepad, thread exiting\n";
            return;
        }

        std::cout << "\n╔════════════════════════════════════════════════╗\n"
                  << "║          XBOX GAMEPAD TELEOP                   ║\n"
                  << "╚════════════════════════════════════════════════╝\n"
                  << "  Movement:  Left stick (forward/back, left/right)\n"
                  << "  Rotation:  Right stick X\n"
                  << "  Mode:      A (stand)  Y (RL control)\n"
                  << "             X (lie down)  B (damping)\n"
                  << "\n";

        float left_x = 0.0f, left_y = 0.0f, right_x = 0.0f;

        while (running_) {
            struct input_event ev;
            int rc = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_NORMAL, &ev);

            while (rc == LIBEVDEV_READ_STATUS_SUCCESS) {
                if (ev.type == EV_ABS) {
                    const struct input_absinfo* info = libevdev_get_abs_info(dev_, ev.code);
                    if (!info) { rc = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_NORMAL, &ev); continue; }

                    float val = NormalizeAxis(ev.value, info);

                    switch (ev.code) {
                        case ABS_X:  left_x = val; break;
                        case ABS_Y:  left_y = val; break;
                        case ABS_RX: right_x = val; break;
                        default: break;
                    }
                }
                else if (ev.type == EV_KEY && ev.value == 1) { // key press only
                    switch (ev.code) {
                        case BTN_SOUTH: // A
                            if (msfb_->GetCurrentState() == RobotMotionState::WaitingForStand
                                || msfb_->GetCurrentState() == RobotMotionState::LieDown) {
                                usr_cmd_->target_mode = uint8_t(RobotMotionState::StandingUp);
                                std::cout << "[MODE] Standing Up\n";
                            }
                            break;
                        case BTN_EAST: // B
                            usr_cmd_->target_mode = uint8_t(RobotMotionState::JointDamping);
                            std::cout << "[MODE] Joint Damping\n";
                            break;
                        case BTN_WEST: // X
                            if (msfb_->GetCurrentState() == RobotMotionState::StandingUp
                                || msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                                usr_cmd_->target_mode = uint8_t(RobotMotionState::LieDown);
                                std::cout << "[MODE] Lie Down\n";
                            }
                            break;
                        case BTN_NORTH: // Y
                            if (msfb_->GetCurrentState() == RobotMotionState::StandingUp) {
                                usr_cmd_->target_mode = uint8_t(RobotMotionState::RLControlMode);
                                std::cout << "[MODE] RL Control\n";
                            }
                            break;
                        default:
                            break;
                    }
                }

                rc = libevdev_next_event(dev_, LIBEVDEV_READ_FLAG_NORMAL, &ev);
            }

            // Update velocity commands
            if (msfb_->GetCurrentState() == RobotMotionState::RLControlMode) {
                float fwd  = -left_y * max_forward_;  // Y axis is inverted on sticks
                float side = -left_x * max_side_;
                float yaw  = -right_x * max_yaw_;
                ClipNumber(fwd, -max_forward_, max_forward_);
                ClipNumber(side, -max_side_, max_side_);
                ClipNumber(yaw, -max_yaw_, max_yaw_);
                usr_cmd_->forward_vel_scale  = fwd;
                usr_cmd_->side_vel_scale     = side;
                usr_cmd_->turnning_vel_scale = yaw;
            } else {
                usr_cmd_->forward_vel_scale  = 0.0f;
                usr_cmd_->side_vel_scale     = 0.0f;
                usr_cmd_->turnning_vel_scale = 0.0f;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }

        CloseDevice();
        std::cout << "\n[XboxGamepad] Stopped.\n";
    }

public:
    XboxGamepadInterface(RobotName robot_name) : UserCommandInterface(robot_name)
    {
        std::cout << "[XboxGamepadInterface] Initialized\n";
        std::memset(usr_cmd_, 0, sizeof(UserCommand));
    }

    ~XboxGamepadInterface()
    {
        Stop();
    }

    void Start() override
    {
        if (running_) return;
        running_ = true;
        gp_thread_ = std::thread(&XboxGamepadInterface::gamepad_loop, this);
    }

    void Stop() override
    {
        running_ = false;
        if (gp_thread_.joinable()) {
            gp_thread_.join();
        }
        usr_cmd_->forward_vel_scale  = 0.0f;
        usr_cmd_->side_vel_scale     = 0.0f;
        usr_cmd_->turnning_vel_scale = 0.0f;
    }

    UserCommand* GetUserCommand() override
    {
        return usr_cmd_;
    }
};
