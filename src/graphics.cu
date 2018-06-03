#ifdef GRAPHICS

#include<GL/glew.h>
#include<GLFW/glfw3.h>
#include<cuda_gl_interop.h>

#include <vector>
#include "headers/helper.h"

// store drawings on screen 
std::vector<Point> draw_points;
std::vector<Point> remove_points; 

// HACK UPDATE FLAG FIELD 
bool obstacles_added = false; 
bool obstacles_removed = false;

GLuint buffer_object;
cudaGraphicsResource *resource;

// process key inputs to close window
void ProcessInput (GLFWwindow* window) {
    // if the escape key has been pressed
    // otherwise glfwGetKey returns GFLW_RELEASE
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose( window, true );
    }
}

// Create frame resize callback function
void FramebufferSizeCallback(GLFWwindow* window, int height, int width) {
    glViewport(0,0, width, height);
}

void MousePressCallback (GLFWwindow* window, int button, int action, int mods) {
    // Use this function to launch other functions when the
    // mouse button is released
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if ( action == GLFW_RELEASE ) {
            // HACK UPDATE FLAG FIELD
            obstacles_added = true;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if ( action == GLFW_RELEASE ) {
            // HACK UPDATE FLAG FIELD
            obstacles_removed = true;
        }
    }
}


void CursorPosCallback(GLFWwindow* window, double x, double y) {
    // store drag positions - check whether button is pressed or not 
    int current_left_button_state = glfwGetMouseButton (window, GLFW_MOUSE_BUTTON_LEFT);
    int current_right_button_state = glfwGetMouseButton (window, GLFW_MOUSE_BUTTON_RIGHT);

    if (current_left_button_state == GLFW_PRESS) {
        draw_points.push_back(Point(x, y)); 
    }

    if (current_right_button_state == GLFW_PRESS) {
        remove_points.push_back(Point(x, y)); 
    }
}
#endif
