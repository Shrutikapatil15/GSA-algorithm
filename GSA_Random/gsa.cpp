#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace std;

struct Agent {
    vector<double> position;
    vector<double> velocity;
};

double random_double(double lb, double ub) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dis(lb, ub);
    return dis(gen);
}

vector<Agent> init_population(int size, int dim, const vector<double>& lb, const vector<double>& ub) {
    vector<Agent> population(size);
    for (int i = 0; i < size; ++i) {
        population[i].position.resize(dim);
        population[i].velocity.resize(dim, 0.0);
        for (int j = 0; j < dim; ++j)
            population[i].position[j] = random_double(lb[j], ub[j]);
    }
    return population;
}

// Objective functions
double sphere_function(const vector<double>& x) {
    double sum = 0.0;
    for (double xi : x)
        sum += xi * xi;
    return sum;
}

double gear_train(const vector<double>& x) {
    return pow((1.0 / 6.931) - ((x[2] * x[1]) / (x[0] * x[3])), 2);
}

double pressure_vessel(const vector<double>& x) {
    double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];
    double g1 = -x1 + 0.0193 * x3;
    double g2 = -x2 + 0.00954 * x3;
    double g3 = 1296000 - ((4.0 / 3.0) * M_PI * pow(x3, 3)) - (M_PI * pow(x3, 2) * x4);
    double g4 = x4 - 240;
    if (g1 <= 0 && g2 <= 0 && g3 <= 0 && g4 <= 0) {
        return 0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 * x3 +
               3.1661 * x1 * x1 * x4 + 19.84 * x1 * x1 * x3;
    } else return 1e10;
}

typedef double (*ObjectiveFunction)(const vector<double>&);

vector<double> trim(const vector<double>& x, const vector<double>& lb, const vector<double>& ub) {
    vector<double> trimmed(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        trimmed[i] = max(lb[i], min(x[i], ub[i]));
    return trimmed;
}

void gsa(vector<Agent>& population, int dim, int iterations, ObjectiveFunction f,
         const vector<double>& lb, const vector<double>& ub,
         double G0, double alpha,
         vector<pair<vector<double>, double>>& iteration_results) {
    
    int N = population.size();

    for (int t = 0; t < iterations; ++t) {
        vector<double> fitness(N);
        for (int i = 0; i < N; ++i)
            fitness[i] = f(population[i].position);

        int best_idx = min_element(fitness.begin(), fitness.end()) - fitness.begin();
        int worst_idx = max_element(fitness.begin(), fitness.end()) - fitness.begin();

        vector<double> best_sol = population[best_idx].position;
        double best_fit = fitness[best_idx];

        cout << "Iteration " << t + 1 << " Best Fitness: " << best_fit << endl;

        iteration_results.push_back({best_sol, best_fit});

        double G = G0 * exp(-alpha * t / (double)iterations);
        vector<double> mass(N);
        double sum_mass = 0.0;

        double best = fitness[best_idx];
        double worst = fitness[worst_idx];
        for (int i = 0; i < N; ++i) {
            mass[i] = (worst - fitness[i]) / (worst - best + 1e-10);
            sum_mass += mass[i];
        }
        for (int i = 0; i < N; ++i)
            mass[i] /= sum_mass;

        for (int i = 0; i < N; ++i) {
            vector<double> force(dim, 0.0);
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double dist = 0.0;
                for (int d = 0; d < dim; ++d)
                    dist += pow(population[j].position[d] - population[i].position[d], 2);
                dist = sqrt(dist) + 1e-10;
                for (int d = 0; d < dim; ++d)
                    force[d] += random_double(0, 1) * (G * mass[i] * mass[j]) / dist *
                                (population[j].position[d] - population[i].position[d]);
            }

            for (int d = 0; d < dim; ++d) {
                double acc = force[d] / (mass[i] + 1e-10);
                population[i].velocity[d] = random_double(0, 1) * population[i].velocity[d] + acc;
                population[i].position[d] += population[i].velocity[d];
            }
            population[i].position = trim(population[i].position, lb, ub);
        }
    }
}

void write_results_to_csv(const vector<pair<vector<double>, double>>& iteration_results,
                          const vector<double>& best_solution, double best_fitness) {
    ofstream file("results.csv");
    if (!file.is_open()) {
        cerr << "Failed to open results.csv\n";
        return;
    }

    int dim = best_solution.size();
    // Write header
    file << "Iteration";
    for (int i = 0; i < dim; ++i)
        file << ",x" << i + 1;
    file << ",Fitness\n";

    for (size_t i = 0; i < iteration_results.size(); ++i) {
        file << i + 1;
        for (double val : iteration_results[i].first)
            file << "," << val;
        file << "," << iteration_results[i].second << "\n";
    }

    // Final best solution
    file << "\nBestSolution";
    for (double val : best_solution)
        file << "," << val;
    file << "," << best_fitness << "\n";

    file.close();
}

int main() {
    int problem;
    cout << "Select problem to solve:\n1. Sphere\n2. Gear Train\n3. Pressure Vessel\n";
    cin >> problem;

    vector<double> lb, ub;
    int dim;

    if (problem == 1) {
        lb = {-100, -100};
        ub = {100, 100};
    } else if (problem == 2) {
        lb = {12, 12, 12, 12};
        ub = {60, 60, 60, 60};
    } else if (problem == 3) {
        lb = {0.0625, 0.0625, 10, 10};
        ub = {1.25, 1.25, 100, 240};
    } else {
        cerr << "Invalid problem.\n";
        return 1;
    }

    dim = lb.size();
    int pop_size, iterations;
    cout << "Enter population size: ";
    cin >> pop_size;
    cout << "Enter number of iterations: ";
    cin >> iterations;

    double G0 = 81.87, alpha = 0.01;
    auto population = init_population(pop_size, dim, lb, ub);
    ObjectiveFunction obj_func = (problem == 1) ? sphere_function :
                                 (problem == 2) ? gear_train :
                                                  pressure_vessel;

    vector<pair<vector<double>, double>> iteration_results;
    gsa(population, dim, iterations, obj_func, lb, ub, G0, alpha, iteration_results);

    auto best = *min_element(iteration_results.begin(), iteration_results.end(),
                             [](const auto& a, const auto& b) {
                                 return a.second < b.second;
                             });

    write_results_to_csv(iteration_results, best.first, best.second);

    cout << "Final Best Fitness: " << best.second << endl;
    return 0;
}
