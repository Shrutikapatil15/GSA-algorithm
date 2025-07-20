#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>

using namespace std;

// Random double in [min, max]
double randDouble(double min, double max) {
    return min + (double)rand() / RAND_MAX * (max - min);
}

// Sphere Function
double sphereFunction(const vector<double>& x) {
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++)
        sum += x[i] * x[i];
    return sum;
}

// Gear Train Function
double gearTrain(const vector<double>& x) {
    return pow((1.0 / 6.931) - (x[2] * x[1]) / (x[0] * x[3]), 2);
}

// Pressure Vessel Function
double pressureVessel(const vector<double>& x) {
    double g1 = -x[0] + 0.0193 * x[2];
    double g2 = -x[1] + 0.00954 * x[2];
    double g3 = 1296000 - (4.0 / 3.0) * M_PI * pow(x[2], 3) - M_PI * pow(x[2], 2) * x[3];
    double g4 = x[3] - 240;
    if (g1 <= 0 && g2 <= 0 && g3 <= 0 && g4 <= 0) {
        return 0.6224 * x[0] * x[2] * x[3] + 1.7781 * x[1] * x[2] * x[2] +
               3.1661 * x[0] * x[0] * x[3] + 19.84 * x[0] * x[0] * x[2];
    }
    return 1e10;
}

// Clamp solution within bounds
void clamp(vector<double>& x, const vector<double>& lb, const vector<double>& ub) {
    for (int i = 0; i < x.size(); i++) {
        if (x[i] < lb[i]) x[i] = lb[i];
        if (x[i] > ub[i]) x[i] = ub[i];
    }
}

// Function pointer type
typedef double (*ObjectiveFunction)(const vector<double>&);

// PSO Algorithm
void pso(int popSize, int dim, int maxIter, ObjectiveFunction objFunc,
         const vector<double>& lb, const vector<double>& ub,
         double w, double c1, double c2) {

    vector<vector<double>> population(popSize, vector<double>(dim));
    vector<vector<double>> velocity(popSize, vector<double>(dim));
    vector<vector<double>> pbest = population;
    vector<double> pbestFitness(popSize, numeric_limits<double>::max());

    // Initialize population and velocities
    for (int i = 0; i < popSize; i++) {
        for (int j = 0; j < dim; j++) {
            population[i][j] = randDouble(lb[j], ub[j]);
            velocity[i][j] = randDouble(-1, 1);
        }
        pbest[i] = population[i];
        pbestFitness[i] = objFunc(population[i]);
    }

    vector<double> gbest = pbest[0];
    double gbestFitness = pbestFitness[0];

    for (int i = 1; i < popSize; i++) {
        if (pbestFitness[i] < gbestFitness) {
            gbest = pbest[i];
            gbestFitness = pbestFitness[i];
        }
    }

    vector<double> fitnessTrack;

    // Main PSO loop
    for (int t = 0; t < maxIter; t++) {
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < dim; j++) {
                double r1 = randDouble(0, 1);
                double r2 = randDouble(0, 1);
                velocity[i][j] = w * velocity[i][j] +
                                 c1 * r1 * (pbest[i][j] - population[i][j]) +
                                 c2 * r2 * (gbest[j] - population[i][j]);
                population[i][j] += velocity[i][j];
            }

            clamp(population[i], lb, ub);
            double fit = objFunc(population[i]);

            if (fit < pbestFitness[i]) {
                pbest[i] = population[i];
                pbestFitness[i] = fit;

                if (fit < gbestFitness) {
                    gbest = population[i];
                    gbestFitness = fit;
                }
            }
        }

        // Track best fitness per iteration
        fitnessTrack.push_back(gbestFitness);
        cout << "Iteration " << t+1 << " - Best Fitness: " << gbestFitness << endl;
    }

    // Print best solution
    cout << "\nBest Solution:\n";
    for (int i = 0; i < gbest.size(); i++)
        cout << "x" << i+1 << " = " << gbest[i] << endl;
    cout << "Best Fitness: " << gbestFitness << endl;

    // Write fitness history to CSV
    ofstream fout("fitness.csv");
    fout << "Iteration,Fitness\n";
    for (int i = 0; i < fitnessTrack.size(); i++) {
        fout << i+1 << "," << fitnessTrack[i] << "\n";
    }
    fout.close();
}

int main() {
    srand(time(0));
    int choice;
    cout << "Select problem:\n1. Sphere\n2. Gear Train\n3. Pressure Vessel\nEnter: ";
    cin >> choice;

    vector<double> lb, ub;
    ObjectiveFunction objFunc;
    int dim;

    if (choice == 1) {
        lb = {-100, -100}; ub = {100, 100}; dim = 2;
        objFunc = sphereFunction;
    } else if (choice == 2) {
        lb = {12, 12, 12, 12}; ub = {60, 60, 60, 60}; dim = 4;
        objFunc = gearTrain;
    } else if (choice == 3) {
        lb = {0, 0, 10, 10}; ub = {100, 100, 200, 200}; dim = 4;
        objFunc = pressureVessel;
    } else {
        cout << "Invalid choice.\n";
        return 1;
    }

    int popSize, maxIter;
    cout << "Enter population size: ";
    cin >> popSize;
    cout << "Enter number of iterations: ";
    cin >> maxIter;

    double w = 0.5, c1 = 1.5, c2 = 1.5;

    pso(popSize, dim, maxIter, objFunc, lb, ub, w, c1, c2);

    return 0;
}
