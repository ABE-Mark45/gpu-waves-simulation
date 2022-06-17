#pragma once
#include <string>
#include <fstream>
#include <iostream>

namespace Utils
{
    void printShaderLog(GLuint shader)
    {
        int len = 0;
        int chWrittn = 0;
        char *log;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        if (len > 0)
        {
            log = (char *)malloc(len);
            glGetShaderInfoLog(shader, len, &chWrittn, log);
            std::cout << "Shader Info Log: " << log << std::endl;
            free(log);
        }
    }

    void printProgramLog(int prog)
    {
        int len = 0;
        int chWrittn = 0;
        char *log;
        glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        if (len > 0)
        {
            log = (char *)malloc(len);
            glGetProgramInfoLog(prog, len, &chWrittn, log);
            std::cout << "Program Info Log: " << log << std::endl;
            free(log);
        }
    }
    bool checkOpenGLError()
    {
        bool foundError = false;
        int glErr = glGetError();
        while (glErr != GL_NO_ERROR)
        {
            std::cout << "glError: " << glErr << std::endl;
            foundError = true;
            glErr = glGetError();
        }
        return foundError;
    }

    std::string readShaderSource(const char *filePath)
    {
        std::string content;
        std::ifstream fileStream(filePath, std::ios::in);
        std::string line = "";
        while (!fileStream.eof())
        {
            getline(fileStream, line);
            content.append(line + "\n");
        }
        fileStream.close();
        return content;
    }

    GLuint createShaderProgram(const char *vp, const char *fp)
    {
        GLint linked;

        GLuint shaderProgram = glCreateProgram();
        GLuint vShader = glCreateShader(GL_VERTEX_SHADER);
        GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);

        std::string vertShaderStr = readShaderSource(vp);
        std::string fragShaderStr = readShaderSource(fp);

        const char *vertShaderSrc = vertShaderStr.c_str();
        const char *fragShaderSrc = fragShaderStr.c_str();

        glShaderSource(vShader, 1, &vertShaderSrc, NULL);
        glShaderSource(fShader, 1, &fragShaderSrc, NULL);

        glCompileShader(vShader);
        checkOpenGLError();

        GLint vertCompiled;
        glGetShaderiv(vShader, GL_COMPILE_STATUS, &vertCompiled);
        if (vertCompiled != 1)
        {
            std::cout << "vertex compilation failed" << std::endl;
            printShaderLog(vShader);
        }

        glCompileShader(fShader);
        checkOpenGLError();

        GLint fragCompiled;
        glGetShaderiv(fShader, GL_COMPILE_STATUS, &fragCompiled);
        if (fragCompiled != 1)
        {
            std::cout << "fragment compilation failed" << std::endl;
            printShaderLog(fShader);
        }
        // catch errors while linking shaders
        glAttachShader(shaderProgram, vShader);
        glAttachShader(shaderProgram, fShader);
        glLinkProgram(shaderProgram);
        checkOpenGLError();
        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &linked);
        if (linked != 1)
        {
            std::cout << "linking failed" << std::endl;
            printProgramLog(shaderProgram);
        }
        return shaderProgram;
    }
}