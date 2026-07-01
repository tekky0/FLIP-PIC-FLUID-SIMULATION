#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <GLFW/glfw3.h>

#define gX 0.0f
#define gY -9.81f
#define r 2.2f
#define h (r*2.0f)
#define particleNUM 100
#define gridX 60
#define gridY 90
#define cellX (int)(gridX/h)
#define cellY (int)(gridY/h)
#define dt (1.0f/30.0f)
#define damping 0.75f
#define visc 0.45f
#define k 2.0f //stiffness constant
#define repulsion .4f
#define alpha .05f //1 means pure flip, 0 means pure pic (flip -> pic)
#define overRelaxation 1.0f
#define rho0 1000
#define epsilon 0.000001
#define kirkleRadiusX 26.0f
#define kirkleRadiusY 39.2f
#define kirkcoordsx ((gridX / 2.0f)-1.2f)
#define kirkcoordsy ((gridY / 2.0f)-1.75f)
#define SPATIAL_CELL_SIZE (2.2f * r)
#define SPATIAL_GRID_X ((int)(gridX / SPATIAL_CELL_SIZE) + 1)
#define SPATIAL_GRID_Y ((int)(gridY / SPATIAL_CELL_SIZE) + 1)

 float* particlePos;
 float* particleVel;

 int* cellType;
 float* u;
 float* v;
 float* pu;
 float* pv;
 float* du;
 float* dv;
 int* s;
 float* divergence;
 float* density;
 float restDensity;

 int* spatialCellCount;
 int* spatialCellStart;
 int* spatialParticleIds;

//static inline float min(float a, float b){
//	if(a > b){
//		return b;
//	}
//	return a;
//}

static inline void spawn_particles() {
	int particlesPerRow = (int)sqrt(particleNUM);
	float space = 1.0f;

	float cubeWidth = particlesPerRow * space;
	float cubeHeight = ceil((float)particleNUM / particlesPerRow) * space;

	float startX = (gridX - cubeWidth) / 2.0f;
	float startY = (gridY - cubeHeight) / 2.0f;

	int index = 0;
	for (int y = 0; index < particleNUM; y++) {
		for (int x = 0; x < particlesPerRow && index < particleNUM; x++) {
			float px = startX + x * space;
			float py = startY + y * space;

			particlePos[index * 2 + 0] = px; // x
			particlePos[index * 2 + 1] = py; // y

			index++;
		}
	}
}

 int cellCount = cellX*cellY;

static inline void allocateMemory() {
// particles
particlePos = (float*)calloc(particleNUM * 2, sizeof(float)); // x,y
particleVel = (float*)calloc(particleNUM * 2, sizeof(float)); // vx,vy

// cells (Nx * Ny grid)
int numSpatialCells = SPATIAL_GRID_X * SPATIAL_GRID_Y;

cellType = (int*)calloc(cellCount, sizeof(int));
u = (float*)calloc(cellCount, sizeof(float));
v = (float*)calloc(cellCount, sizeof(float));
pu = (float*)calloc(cellCount, sizeof(float));
pv = (float*)calloc(cellCount, sizeof(float));
du = (float*)calloc(cellCount, sizeof(float));
dv = (float*)calloc(cellCount, sizeof(float));
s = (int*)calloc(cellCount, sizeof(float));
divergence = (float*)calloc(cellCount, sizeof(float)); // Updated variable name
density = calloc(cellCount, sizeof(float));
for (int i = 0; i < cellCount; i++) s[i] = 1;

spatialCellStart = (int*)calloc(numSpatialCells+1, sizeof(int));
spatialParticleIds = calloc(particleNUM, sizeof(int));
spatialCellCount = calloc(numSpatialCells, sizeof(int));
//spawnParticlesSquare(gridX * 0.5f, gridY * 0.5f, 40.0f);
}

static inline void reset_Memory() {
	for (int i = 0; i < cellCount; i++) {
		density[i] = 0.0f;

	}
}


static inline void integrateParticles(int integrate) {
	for (int i = 0; i < particleNUM; i++) {
		// Apply gravity to velocity
		if (integrate) {
			particleVel[2 * i] += gX * dt;
			particleVel[2 * i + 1] += gY * dt;

			// Update positions
			particlePos[2 * i] += particleVel[2 * i] * dt;
			particlePos[2 * i + 1] += particleVel[2 * i + 1] * dt;
		}

		float* x = &particlePos[i * 2];
		float* y = &particlePos[i * 2 + 1];
		float* vx = &particleVel[i * 2];
		float* vy = &particleVel[i * 2 + 1];
		if (*x < 0) { *x = 0;      *vx *= -damping; }
		if (*x > gridX) { *x = gridX;  *vx *= -damping; }
		if (*y < 0) { *y = 0;      *vy *= -damping; }
		if (*y > gridY) { *y = gridY;  *vy *= -damping; }

		// FIXED: Correct circle center and radius
		float cx = kirkcoordsx;
		float cy = kirkcoordsy;
		float Rx = kirkleRadiusX;
		float Ry = kirkleRadiusY;

		float dx = *x - cx;
		float dy = *y - cy;

		// Ellipse equation: (dx/Rx)^2 + (dy/Ry)^2 > 1
		float normDist2 = (dx / Rx) * (dx / Rx) + (dy / Ry) * (dy / Ry);

		if (normDist2 > 1.0f) {
			// Ellipse normal (gradient of the ellipse equation)
			float nx = dx / (Rx * Rx);
			float ny = dy / (Ry * Ry);
			float len = sqrtf(nx * nx + ny * ny);
			nx /= len;
			ny /= len;

			float vn = (*vx) * nx + (*vy) * ny;
			if (vn > 0.0f) {
				*vx -= vn * nx * damping;
				*vy -= vn * ny * damping;
			}

			// Push back to ellipse surface
			// Approximate: scale position back along the normal
			float scale = 1.0f / sqrtf(normDist2);
			*x = cx + dx * scale;
			*y = cy + dy * scale;
		}
	}
}



static inline float clamp(float x, float minVal, float maxVal) {
	if (x < minVal) return minVal;
	if (x > maxVal) return maxVal;
	return x;
}

static inline void pushParticlesApart(int iter_) {
	float minDist = 2.0f * r;
	float minDist2 = minDist * minDist;

	int spatialGridX = SPATIAL_GRID_X;
	int spatialGridY = SPATIAL_GRID_Y;
	int numSpatialCells = spatialGridX * spatialGridY;

	for (int iter = 0; iter < iter_; iter++) {
		// Reset cell counts
		memset(spatialCellCount, 0, numSpatialCells * sizeof(int));

		// Count particles per cell
		for (int i = 0; i < particleNUM; i++) {
			float x = particlePos[2 * i];
			float y = particlePos[2 * i + 1];

			int xi = (int)(x / SPATIAL_CELL_SIZE);
			int yi = (int)(y / SPATIAL_CELL_SIZE);
			xi = clamp(xi, 0, spatialGridX - 1);
			yi = clamp(yi, 0, spatialGridY - 1);

			int cellIdx = xi * spatialGridY + yi;
			spatialCellCount[cellIdx]++;
		}


		// Build prefix sum
		int sum = 0;
		for (int i = 0; i < numSpatialCells; i++) {
			spatialCellStart[i] = sum;               // start of bucket
			sum += spatialCellCount[i];
		}
		spatialCellStart[numSpatialCells] = sum;

		memset(spatialCellCount, 0, numSpatialCells * sizeof(int));

		for (int i = 0; i < particleNUM; i++) {
			float x = particlePos[2 * i];
			float y = particlePos[2 * i + 1];

			int xi = (int)(x / SPATIAL_CELL_SIZE);
			int yi = (int)(y / SPATIAL_CELL_SIZE);
			xi = clamp(xi, 0, spatialGridX - 1);
			yi = clamp(yi, 0, spatialGridY - 1);

			int cellIdx = xi * spatialGridY + yi;
			int index = spatialCellStart[cellIdx] + spatialCellCount[cellIdx]++;
			spatialParticleIds[index] = i;
			//spatialCellCount[cellIdx]++;
		}

		for (int i = 0; i < particleNUM; i++) {
			float px = particlePos[2 * i];
			float py = particlePos[2 * i + 1];

			int pxi = (int)(px / SPATIAL_CELL_SIZE);
			int pyi = (int)(py / SPATIAL_CELL_SIZE);

			// Check 3x3 neighborhood
			for (int dx = -1; dx <= 1; dx++) {
				for (int dy = -1; dy <= 1; dy++) {
					int xi = pxi + dx;
					int yi = pyi + dy;

					if (xi < 0 || xi >= spatialGridX || yi < 0 || yi >= spatialGridY) continue;

					int cellIdx = xi * spatialGridY + yi;
					int first = spatialCellStart[cellIdx];
					int last = first + spatialCellCount[cellIdx];

					for (int j = first; j < last; j++) {
						int id = spatialParticleIds[j];
						if (id == i) continue;

						float qx = particlePos[2 * id];
						float qy = particlePos[2 * id + 1];

						float dx = qx - px;
						float dy = qy - py;
						float d2 = dx * dx + dy * dy;

						if (d2 > minDist2 || d2 == 0.0f) continue;

						float d = sqrtf(d2);
						float s = repulsion * (minDist - d) / d;
						dx *= s;
						dy *= s;
						//if (id <= solid_Particles) {
						//    particleVel[2 * i] *= -.1f;
						//    particleVel[2 * i + 1] *= -.1f;
						//}
						particlePos[2 * i] -= dx;
						particlePos[2 * i + 1] -= dy;
						particlePos[2 * id] += dx;
						particlePos[2 * id + 1] += dy;
					}
				}
			}
		}
	}
}

//now we compute cell-particle density as rho
//lazy right now ill do transfer velocities and solve at a later date
static inline void computeDensity() {
	for (int den = 0; den < cellCount; den++) {
		density[den] = 0.0f;
	}
	float h1 = 1.0f / h;
	float h2 = 0.5f * h;

	for (int i = 0; i < particleNUM; i++) {
		float x = clamp(particlePos[i * 2], h, (cellX-1)*h);
		float y = clamp(particlePos[i * 2 + 1], h, (cellY - 1) * h);

		int x0 = (int)((x - h2) * h1);
		float tx = ((x - h2) - x0 * h) * h1;
		int x1 = (int)min(x0 + 1, cellX - 2);

		int y0 = (int)((y - h2) * h1);
		float ty = ((y - h2) - y0 * h) * h1;
		int y1 = (int)min(y0 + 1, cellY - 2);

		float sx = 1.0f - tx;
		float sy = 1.0f - ty;

		if (x0 < cellX && y0 < cellY) density[x0 * cellY + y0] += sx * sy;
		if (x1 < cellX && y0 < cellY) density[x1 * cellY + y0] += tx * sy;
		if (x1 < cellX && y1 < cellY) density[x1 * cellY + y1] += tx * ty;
		if (x0 < cellX && y1 < cellY) density[x0 * cellY + y1] += sx * ty;
	}

	if (restDensity == 0.0f) {
		float sum = 0.0f;
		int numFluidCells = 0;
		int numCells = cellX * cellY;
		for (int cell = 0; cell < numCells; cell++) {
			if (cellType[cell] == 2) {
				sum += density[cell]; //if fluid compute density sum of cell;
				numFluidCells++;
			}
		}

		if (numFluidCells > 0) {
			restDensity = sum / numFluidCells;
		}
	}
}

static inline void transferVelocity(int toGrid) {
	int ny = cellY;
	int nx = cellX;
	float h1 = 1.0f / h;
	float h2 = 0.5f * h;

	//reset cell
	if (toGrid) {
		memcpy(pu, u, sizeof(float) * cellCount);
		memcpy(pv, v, sizeof(float) * cellCount);
		for (int res = 0; res < cellCount; res++) {
			u[res] = 0.0f;
			v[res] = 0.0f;
			du[res] = 0.0f;
			dv[res] = 0.0f;
		}


		for (int i = 0; i < cellCount; i++) {
			cellType[i] = s[i] == 0 ? 0 : 1; //solid : air
		}

		for (int j = 0; j < particleNUM; j++) {
			float x = particlePos[j * 2];
			float y = particlePos[j * 2 + 1];
			int xi = (int)clamp(floor(x*h1),0.0f, nx - 1);
			int yi = (int)clamp(floor(y*h1),0.0f, ny - 1);
			int index = xi * ny + yi;
			if (cellType[index] == 1) cellType[index] = 2; // if air, make fluid type
		}
	}


	for (int comp = 0; comp < 2; comp++) {
		float dx = comp == 0 ? 0.0f : h2;
		float dy = comp == 0 ? h2 : 0.0f;

		float* f = comp == 0 ? u : v;
		float* prevF = comp == 0 ? pu : pv;
		float* d = comp == 0 ? du : dv;

		//now we do grid to particles
		//find 4 cells
		for (int p = 0; p < particleNUM; p++) {
			float x = particlePos[p * 2];
			float y = particlePos[p * 2 + 1];

			x = clamp(x, h, (float)((nx - 1) * h));
			y = clamp(y, h, (float)((ny - 1) * h));

			int x0 = (int)clamp(floorf(x - dx), 0.0f, (float)((nx - 2)));
			int y0 = (int)clamp(floorf(y - dy), 0.0f, (float)((ny - 2)));
			//now we have cell coords

			//locate neighbor x
			//locate right and top cells
			int x1 = (int)min(x0 + 1, cellX - 2);
			int y1 = (int)min(y0 + 1, cellY - 2);

			//compensate stagger
			float tx = ((x - dx) - x0 * h) * h1;
			float ty = ((y - dy) - y0 * h) * h1;

			float sx = 1.0f - tx;
			float sy = 1.0f - ty;
			// compute weights

			float w0 = sx * sy;
			float w1 = tx * sy;
			float w2 = tx * ty;
			float w3 = sx * ty;

			int nr0 = x0 * cellY + y0;
			int nr1 = x1 * cellY + y0;
			int nr2 = x1 * cellY + y1;
			int nr3 = x0 * cellY + y1;


			if (toGrid) {
				float pv = particleVel[2 * p + comp];
				f[nr0] += pv * w0; d[nr0] += w0;
				f[nr1] += pv * w1; d[nr1] += w1;
				f[nr2] += pv * w2; d[nr2] += w2;
				f[nr3] += pv * w3; d[nr3] += w3;
			}
			else {
				// G2P transfer
				int offset = comp == 0 ? cellY : 1;
				//float f0 = ((cellType[nr0] != 1) || cellType[nr0 - offset] != 1) ? 1.0f : 0.0f;
				//float f1 = ((cellType[nr1] != 1) || cellType[nr1 - offset] != 1) ? 1.0f : 0.0f;
				//float f2 = ((cellType[nr2] != 1) || cellType[nr2 - offset] != 1) ? 1.0f : 0.0f;
				//float f3 = ((cellType[nr3] != 1) || cellType[nr3 - offset] != 1) ? 1.0f : 0.0f;
				float f0 = (cellType[nr0] != 1) ? 1.0f : 0.0f;
				float f1 = (cellType[nr1] != 1) ? 1.0f : 0.0f;
				float f2 = (cellType[nr2] != 1) ? 1.0f : 0.0f;
				float f3 = (cellType[nr3] != 1) ? 1.0f : 0.0f;
				float d = f0 * w0 + f1 * w1 + f2 * w2 + f3 * w3;
				float vel = particleVel[p * 2 + comp];


				// blend FLIP and PIC
				//particleVel[2 * p + comp] = (1.0f - alpha) * flip + alpha * pic;
				if (d > 0.0f) {
					float pic = (f0 * w0 * f[nr0] + f1*w1*f[nr1] + f2 * w2 * f[nr2] + f3 * w3 * f[nr3])/d;
					float corr = (
						(f0 * w0 * (f[nr0] - prevF[nr0])) +
						(f1 * w1 * (f[nr1] - prevF[nr1])) +
						(f2 * w2 * (f[nr2] - prevF[nr2])) +
						(f3 * w3 * (f[nr3] - prevF[nr3]))
						)/d;
					float flip = vel + corr;
					particleVel[2 * p + comp] = alpha * flip + (1.0f - alpha) * pic;
				}
			}
		}
		if (toGrid) {
			for (int i = 0; i < cellCount; i++) {
				if (d[i] > 0.0f) {
					f[i] /= d[i];
				}
			}
			for (int i = 0; i < cellX; i++) {
				for (int j = 0; j < cellY; j++) {
					int solid = cellType[i * cellY + j];
					if (comp == 0) {
						if (solid || (i > 0 && cellType[(i - 1) * cellY + j] == 0))
							u[i * cellY + j] = pu[i * cellY + j];
					}
					else {
						if (solid || (j > 0 && cellType[i * cellY + j - 1] == 0))
							v[i * cellY + j] = pv[i * cellY + j];
					}

				}
			}
		}
	}
}

static inline void solveIncompressibility(int numIter) {
	memset(divergence, 0.0f, cellCount * sizeof(float));
	memcpy(pu, u, cellCount * sizeof(float));
	memcpy(pv, v, cellCount * sizeof(float));
	//reset divergence array and clone the previous velocity components for differences later
	float cp = rho0 * h / dt;
	//run based on user defined divergence/pressure solve iterations
	for (int iter = 0; iter < numIter; iter++) {
		for (int i = 1; i < cellX - 1; i++) {
			for (int j = 1; j < cellY - 1; j++) {
				if (cellType[i * cellY + j] == 0) continue;

				int center = i * cellY + j;
				int left = (i - 1) * cellY + j;
				int right = (i + 1) * cellY + j;
				int top = i * cellY + j + 1;
				int bottom = i * cellY + j - 1;
				//defined direct neighbors from center;
				  int c = i*cellY + j;
				//float visc = 0.1f;
				u[c] += visc * (u[c - 1] + u[c + 1] + u[c - cellY] + u[c + cellY] - 4.0f * u[c]);
				v[c] += visc * (v[c - 1] + v[c + 1] + v[c - cellY] + v[c + cellY] - 4.0f * v[c]);

				int sc = s[center];
				int sl = s[left];
				int sr = s[right];
				int st = s[top];
				int sb = s[bottom];
				int sValidNum = sl + sr + st + sb;
				if (sValidNum == 0) continue;
				//validity


				//boundary solid
				u[right] = (cellType[right] != 0) ? u[right] : 0.0f;
				u[center] = (cellType[center] != 0) ? u[center] : 0.0f;
				v[top] = (cellType[top] != 0) ? v[top] : 0.0f;
				v[center] = (cellType[center] != 0) ? v[center] : 0.0f;


				//solve for divergence;
				float div = u[right] - u[center] + v[top] - v[center];

				if (restDensity > 0.0f) {
					float compression = density[i * cellY + j] - restDensity;
					if (compression > 0.0f) {
						div -= k * compression;
					}
				}

				float p = (-div / sValidNum)*overRelaxation;
				divergence[center] += cp * p;
				u[center] -= sl * p;
				u[right] += sr * p;
				v[top] += st * p;
				v[center] -= sb * p;



			}
		}
	}
}


static inline void setSolidCell(int i, int j) {
	int idx = i * cellY + j;
	cellType[idx] = 0;
	s[idx] = 0.0;  // make sure "validity" array says no fluid passes through
	//float x = (float)i + (h/2.0f);
	//float y = (float)j + (h/2.0f);
	//particlePos[n * 2] = x;
	//particlePos[n * 2 + 1] = y;
	//particleVel[n * 2] = 0.0f;
	//particleVel[n * 2 + 1] = 0.0f;
}

static inline void setBoundaryWalls() {
	for (int i = 0; i < cellX; i++) {
		for (int j = 0; j < cellY; j++) {
			if (i == 0 || j == 0 || i == cellX - 1 || j == cellY - 1) {
				setSolidCell(i, j);
			}
		}
	}
}



int main() {
    allocateMemory();
    spawn_particles();
    //init + version declares
    GLFWwindow* window;
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    window = glfwCreateWindow(800, 800, "Fluid Sim", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // --- OpenGL 2D setup ---
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black background
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    int count = 0;
    setBoundaryWalls();
    double lastTime = glfwGetTime();
    while (!glfwWindowShouldClose(window)) {
        //glfwPollEvents();
        //logic
        double ctime = glfwGetTime();
        double deltaTime = ctime - lastTime;
        lastTime = ctime;
        printf("frame: %d\ntime: %.2f\nframe/sec: %.2f\n", count++, ctime, count / ctime);
        integrateParticles(1);
        pushParticlesApart(17);
        integrateParticles(0);
        transferVelocity(1);
        computeDensity();
        solveIncompressibility(12);
        transferVelocity(0);

        //logic


        //boundary / collisions
        //boundary / collisions

       // --- Rendering ---
        glClear(GL_COLOR_BUFFER_BIT);
        //renderGrid();
        glLoadIdentity();
        //glColor3f(1.0f, 1.0f, 1.0f); // White particles
        glPointSize(3.5f);

        // In your rendering code
        glBegin(GL_POINTS);
        for (int i = 0; i < particleNUM; i++) {
            glColor3f(0.0f, 1.0f, 0.0f);

            float x = particlePos[i * 2];
            float y = particlePos[i * 2 + 1];
            float nx = (x / gridX) * 2.0f - 1.0f;
            float ny = (y / gridY) * 2.0f - 1.0f;

            glVertex2f(nx, ny);
        }
        glEnd();
        glfwSwapBuffers(window);
        //printf("problem\n");
        glfwPollEvents();
        //drawParticles();
        //glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;

}