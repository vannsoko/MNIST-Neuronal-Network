#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <algorithm>
#include <cassert>
#include <cstring>

class NeuralNetwork {
private:
    std::vector<int> layers;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> z_values;
    double learning_rate;
    std::mt19937 rng;

    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }

    // Initialize weights and biases with random values
    void initialize_weights() {
        std::normal_distribution<double> dist(0.0, 1.0);
        
        weights.resize(layers.size() - 1);
        biases.resize(layers.size() - 1);
        
        for (int i = 0; i < layers.size() - 1; i++) {
            weights[i].resize(layers[i + 1]);
            biases[i].resize(layers[i + 1]);
            
            for (int j = 0; j < layers[i + 1]; j++) {
                weights[i][j].resize(layers[i]);
                biases[i][j] = dist(rng);
                
                for (int k = 0; k < layers[i]; k++) {
                    weights[i][j][k] = dist(rng) / std::sqrt(layers[i]);
                }
            }
        }
    }

public:
    NeuralNetwork(std::vector<int> layer_sizes, double lr = 0.1) 
        : layers(layer_sizes), learning_rate(lr), rng(std::random_device{}()) {
        initialize_weights();
        
        // Initialize activation vectors
        activations.resize(layers.size());
        z_values.resize(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            activations[i].resize(layers[i]);
            z_values[i].resize(layers[i]);
        }
    }

    // Forward propagation
    std::vector<double> forward(const std::vector<double>& input) {
        assert(input.size() == layers[0]);
        
        activations[0] = input;
        
        // Propagate through hidden and output layers
        for (int i = 1; i < layers.size(); i++) {
            for (int j = 0; j < layers[i]; j++) {
                z_values[i][j] = biases[i-1][j];
                
                for (int k = 0; k < layers[i-1]; k++) {
                    z_values[i][j] += weights[i-1][j][k] * activations[i-1][k];
                }
                
                activations[i][j] = sigmoid(z_values[i][j]);
            }
        }
        
        return activations.back();
    }

    void backward(const std::vector<double>& target) {
        int num_layers = layers.size();
        std::vector<std::vector<double>> errors(num_layers);
        

        for (int i = 0; i < num_layers; i++) {
            errors[i].resize(layers[i]);
        }
        

        for (int i = 0; i < layers[num_layers - 1]; i++) {
            double output_error = target[i] - activations[num_layers - 1][i];
            errors[num_layers - 1][i] = output_error * sigmoid_derivative(z_values[num_layers - 1][i]);
        }

        for (int layer = num_layers - 2; layer >= 1; layer--) {
            for (int i = 0; i < layers[layer]; i++) {
                double error_sum = 0.0;
                for (int j = 0; j < layers[layer + 1]; j++) {
                    error_sum += weights[layer][j][i] * errors[layer + 1][j];
                }
                errors[layer][i] = error_sum * sigmoid_derivative(z_values[layer][i]);
            }
        }
        

        for (int layer = 0; layer < num_layers - 1; layer++) {
            for (int i = 0; i < layers[layer + 1]; i++) {

                biases[layer][i] += learning_rate * errors[layer + 1][i];
                

                for (int j = 0; j < layers[layer]; j++) {
                    weights[layer][i][j] += learning_rate * errors[layer + 1][i] * activations[layer][j];
                }
            }
        }
    }

    // Train the network on a single sample
    void train_sample(const std::vector<double>& input, const std::vector<double>& target) {
        forward(input);
        backward(target);
    }

    // Calculate loss (mean squared error)
    double calculate_loss(const std::vector<double>& target) {
        double loss = 0.0;
        std::vector<double> output = activations.back();
        
        for (int i = 0; i < output.size(); i++) {
            double diff = target[i] - output[i];
            loss += diff * diff;
        }
        
        return loss / output.size();
    }

    // Get prediction (index of highest output)
    int predict(const std::vector<double>& input) {
        std::vector<double> output = forward(input);
        return std::max_element(output.begin(), output.end()) - output.begin();
    }

    // Test accuracy on a dataset
    double test_accuracy(const std::vector<std::vector<double>>& test_inputs,
                        const std::vector<int>& test_labels) {
        int correct = 0;
        
        for (int i = 0; i < test_inputs.size(); i++) {
            int prediction = predict(test_inputs[i]);
            if (prediction == test_labels[i]) {
                correct++;
            }
        }
        
        return static_cast<double>(correct) / test_inputs.size();
    }

    // Train the network
    void train(const std::vector<std::vector<double>>& train_inputs,
              const std::vector<std::vector<double>>& train_targets,
              int epochs) {
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            double total_loss = 0.0;
            
            // Shuffle training data
            std::vector<int> indices(train_inputs.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), rng);
            
            // Train on each sample
            for (int idx : indices) {
                train_sample(train_inputs[idx], train_targets[idx]);
                forward(train_inputs[idx]);
                total_loss += calculate_loss(train_targets[idx]);
            }
            
            if (epoch % 10 == 0) {
                std::cout << "Epoch " << epoch << ", Average Loss: " 
                         << total_loss / train_inputs.size() << std::endl;
            }
        }
    }
};

// Helper function to convert label to one-hot encoding
std::vector<double> to_one_hot(int label, int num_classes = 10) {
    std::vector<double> one_hot(num_classes, 0.0);
    one_hot[label] = 1.0;
    return one_hot;
}

// Utility function to reverse bytes (MNIST files are big-endian)
uint32_t reverse_bytes(uint32_t value) {
    return ((value & 0xFF000000) >> 24) |
           ((value & 0x00FF0000) >> 8)  |
           ((value & 0x0000FF00) << 8)  |
           ((value & 0x000000FF) << 24);
}

// MNIST Data Loader Class
class MNISTLoader {
public:
    std::vector<std::vector<double>> images;
    std::vector<int> labels;
    
    // Load MNIST images from binary file
    bool load_images(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            std::cerr << "Expected files:" << std::endl;
            std::cerr << "  - train-images-idx3-ubyte (training images)" << std::endl;
            std::cerr << "  - train-labels-idx1-ubyte (training labels)" << std::endl;
            std::cerr << "  - t10k-images-idx3-ubyte (test images)" << std::endl;
            std::cerr << "  - t10k-labels-idx1-ubyte (test labels)" << std::endl;
            return false;
        }
        
        // Read header
        uint32_t magic, num_images, rows, cols;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_images), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);
        
        // Convert from big-endian
        magic = reverse_bytes(magic);
        num_images = reverse_bytes(num_images);
        rows = reverse_bytes(rows);
        cols = reverse_bytes(cols);
        
        std::cout << "Loading " << num_images << " images of size " 
                  << rows << "x" << cols << std::endl;
        
        if (magic != 2051) {
            std::cerr << "Error: Invalid magic number in image file" << std::endl;
            return false;
        }
        
        // Read images
        images.resize(num_images);
        for (uint32_t i = 0; i < num_images; i++) {
            images[i].resize(rows * cols);
            std::vector<unsigned char> buffer(rows * cols);
            file.read(reinterpret_cast<char*>(buffer.data()), rows * cols);
            
            // Convert to double and normalize to [0,1]
            for (int j = 0; j < rows * cols; j++) {
                images[i][j] = static_cast<double>(buffer[j]) / 255.0;
            }
        }
        
        file.close();
        return true;
    }
    
    // Load MNIST labels from binary file
    bool load_labels(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return false;
        }
        
        // Read header
        uint32_t magic, num_labels;
        file.read(reinterpret_cast<char*>(&magic), 4);
        file.read(reinterpret_cast<char*>(&num_labels), 4);
        
        // Convert from big-endian
        magic = reverse_bytes(magic);
        num_labels = reverse_bytes(num_labels);
        
        std::cout << "Loading " << num_labels << " labels" << std::endl;
        
        if (magic != 2049) {
            std::cerr << "Error: Invalid magic number in label file" << std::endl;
            return false;
        }
        
        // Read labels
        labels.resize(num_labels);
        std::vector<unsigned char> buffer(num_labels);
        file.read(reinterpret_cast<char*>(buffer.data()), num_labels);
        
        for (uint32_t i = 0; i < num_labels; i++) {
            labels[i] = static_cast<int>(buffer[i]);
        }
        
        file.close();
        return true;
    }
    
    // Load both images and labels
    bool load_data(const std::string& image_file, const std::string& label_file) {
        return load_images(image_file) && load_labels(label_file);
    }
    
    // Get a subset of the data
    void get_subset(int start, int count, 
                   std::vector<std::vector<double>>& subset_images,
                   std::vector<int>& subset_labels) {
        int end = std::min(start + count, static_cast<int>(images.size()));
        
        subset_images.clear();
        subset_labels.clear();
        
        for (int i = start; i < end; i++) {
            subset_images.push_back(images[i]);
            subset_labels.push_back(labels[i]);
        }
    }
    
    // Print a digit as ASCII art (for visualization)
    void print_digit(int index) {
        if (index >= images.size()) {
            std::cerr << "Index out of range" << std::endl;
            return;
        }
        
        std::cout << "Label: " << labels[index] << std::endl;
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double pixel = images[index][i * 28 + j];
                if (pixel > 0.5) std::cout << "##";
                else if (pixel > 0.2) std::cout << "..";
                else std::cout << "  ";
            }
            std::cout << std::endl;
        }
    }
};

// Example usage with real MNIST data
int main() {
    std::cout << "MNIST Neural Network\n";
    std::cout << "====================\n";
    
    // Try to load real MNIST data first
    MNISTLoader mnist;
    
    std::cout << "Loading MNIST data files...\n";
    if (mnist.load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte")) {
        std::cout << "Successfully loaded MNIST data!\n";
    } else {
        std::cout << "Could not load MNIST data files. Using dummy data for demonstration.\n";
        std::cout << "\nTo use real data, download MNIST files from:\n";
        std::cout << "http://yann.lecun.com/exdb/mnist/\n";
        std::cout << "and place them in the same directory as this program.\n\n";
    }
    
    // Network architecture
    std::vector<int> architecture = {784, 128, 64, 10};
    NeuralNetwork network(architecture, 0.1);
    
    std::cout << "Created neural network with architecture: ";
    for (int i = 0; i < architecture.size(); i++) {
        std::cout << architecture[i];
        if (i < architecture.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    
    std::vector<std::vector<double>> train_inputs;
    std::vector<std::vector<double>> train_targets;
    std::vector<int> train_labels;
    
    std::vector<std::vector<double>> subset_images;
    std::vector<int> subset_labels;
    mnist.get_subset(0, 5000, subset_images, subset_labels);
    
    train_inputs = subset_images;
    train_labels = subset_labels;
    
    // Convert labels to one-hot encoding
    for (int label : subset_labels) {
        train_targets.push_back(to_one_hot(label));
    }
    
    std::cout << "Using " << train_inputs.size() << " MNIST samples\n";
    
    
    // Train the network
    std::cout << "\nTraining network...\n";
    int epochs = 50;
    network.train(train_inputs, train_targets, epochs);
    
    // Test the network
    std::cout << "\nTesting network...\n";
    double accuracy = network.test_accuracy(train_inputs, train_labels);
    std::cout << "Training accuracy: " << accuracy * 100 << "%\n";
    
    if (mnist.images.size() > 5000) {
        std::cout << "\nTesting on additional samples...\n";
        
        std::vector<std::vector<double>> test_images;
        std::vector<int> test_labels;
        mnist.get_subset(5000, 1000, test_images, test_labels);
        
        double test_accuracy = network.test_accuracy(test_images, test_labels);
        std::cout << "Test accuracy on new samples: " << test_accuracy * 100 << "%\n";
        
        // Show some predictions
        std::cout << "\nSample predictions:\n";
        for (int i = 0; i < 5; i++) {
            int prediction = network.predict(test_images[i]);
            std::cout << "Sample " << i << " - Predicted: " << prediction 
                     << ", Actual: " << test_labels[i] 
                     << (prediction == test_labels[i] ? " ✓" : " ✗") << std::endl;
        }
    }
    
    return 0;
}