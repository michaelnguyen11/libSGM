
#ifndef _WITROBOT_IMU_PARSER_HPP__
#define _WITROBOT_IMU_PARSER_HPP__

#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <sstream>
#include <fstream>

#include <eigen3/Eigen/Dense>

namespace dns
{
namespace camera
{

struct ImuData
{
    uint64_t timestamp_ms;

    Eigen::Vector3d angularVelocity;
    // Eigen::Vector3d magnetometer;
    Eigen::Vector3d linearAcceleration;
    Eigen::Vector4d quaternion;
};

enum IMU_Type
{
    IMU_Type_UNKNOW = 0,
    IMU_Type_AMGQUA
};

class IMU_Parser
{

public:
    IMU_Parser() {}
    ~IMU_Parser() {}

    static bool parse(const std::string &str, ImuData &outdata)
    {
        // std::cout << "IMU rawdata: " << str << std::endl;
        std::vector<std::string> splitImuData = IMU_Parser::string_splitter(str, ',');

        if (splitImuData.size() == 0)
        {
            return false;
        }

        bool ret = true;

        if (splitImuData[0] == "$AMGQUA")
        {
            if (splitImuData.size() >= 15)
            {
                try
                {
                    outdata.timestamp_ms = std::stoul(splitImuData[1]);
                    // original data used the g unit, convert to m/s^2
                    outdata.linearAcceleration[0] = std::stod(splitImuData[2]) / 100.;
                    outdata.linearAcceleration[1] = std::stod(splitImuData[3]) / 100.;
                    outdata.linearAcceleration[2] = std::stod(splitImuData[4]) / 100.;

                    // original data used the uTesla unit, convert to Tesla
                    // for (int i = 5; i <= 7; ++i)
                    //     magnetometer[i] = (std::stod(splitImuData[i])/16.);

                    // original data used the degree/s unit, convert to radian/s
                    outdata.angularVelocity[0] = std::stod(splitImuData[8]) / 900.;
                    outdata.angularVelocity[1] = std::stod(splitImuData[9]) / 900.;
                    outdata.angularVelocity[2] = std::stod(splitImuData[10]) / 900.;

                    // Orientation
                    outdata.quaternion[0] = std::stod(splitImuData[11]) / 16384.;
                    outdata.quaternion[1] = std::stod(splitImuData[12]) / 16384.;
                    outdata.quaternion[2] = std::stod(splitImuData[13]) / 16384.;
                    outdata.quaternion[3] = std::stod(splitImuData[14]) / 16384.;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "IMU_AMGQUA: " << e.what() << '\n';
                }
            }
            else
            {
                ret = false;
            }
        }

        return true;
    }

    static std::vector<std::string> string_splitter(const std::string &data, const char &delimiter)
    {

        std::vector<std::string> internal;
        std::stringstream ss(data);
        std::string token;

        while (std::getline(ss, token, delimiter))
        {
            internal.push_back(token);
        }
        return internal;
    }
};

class IMU_Reader
{

public:
    typedef std::shared_ptr<IMU_Reader> Ptr;

    explicit IMU_Reader(const std::string &filename, IMU_Type defaultType = IMU_Type_AMGQUA)
    {
        _imuFileReader.open(filename);
        if (_imuFileReader.is_open())
        {
            std::string cmd;
            switch (defaultType)
            {
            case IMU_Type_AMGQUA:
            {
                cmd = "echo @AMGQUA >" + filename;
                break;
            }
            default:
                break;
            }

            if (!cmd.empty())
            {
                system(cmd.c_str());
            }
        }
    }

    ~IMU_Reader()
    {
        if (_imuFileReader.is_open())
            _imuFileReader.close();
    }

    bool isOpen()
    {
        return _imuFileReader.is_open();
    }

    bool fetch_imu(ImuData &outdata)
    {
        if (_imuFileReader.is_open())
        {
            std::string data;
            _imuFileReader >> data;
            return IMU_Parser::parse(data, outdata);
        }
        else
        {
            return false;
        }
    }

private:
    std::fstream _imuFileReader;
};

} // namespace camera
} // namespace dns

#endif