#ifndef SEQUENTIAL_LBM_SRC_HEADERS_GRAPHICS_H_
#define SEQUENTIAL_LBM_SRC_HEADERS_GRAPHICS_H_

// store drawings on screen 
extern std::vector<Point> draw_points;
extern std::vector<Point> remove_points; 

// HACK UPDATE FLAG FIELD 
extern bool obstacles_added; 
extern bool obstacles_removed;
extern GLuint buffer_object;
extern cudaGraphicsResource *resource;

void ProcessInput(GLFWwindow* window);
void FramebufferSizeCallback(GLFWwindow* window, int height, int width);
void MousePressCallback(GLFWwindow* window, int button, int action, int mods);
void CursorPosCallback(GLFWwindow* window, double x, double y);

#endif  // SEQUENTIAL_LBM_SRC_HEADERS_GRAPHICS_H_
