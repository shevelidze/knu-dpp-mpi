#include <mpi.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <queue>
#include <map>

struct HuffmanTreeNode {
  char ch;
  int freq;
  HuffmanTreeNode* left;
  HuffmanTreeNode* right;
  HuffmanTreeNode(char ch, int freq)
      : ch(ch), freq(freq), left(nullptr), right(nullptr) {}
  HuffmanTreeNode(char ch, int freq, HuffmanTreeNode* left, HuffmanTreeNode* right)
      : ch(ch), freq(freq), left(left), right(right) {}
};

struct compare {
  bool operator()(HuffmanTreeNode* l, HuffmanTreeNode* r) {
    return l->freq > r->freq;
  }
};

void generateCodes(HuffmanTreeNode* root, std::string str,
                   std::unordered_map<char, std::string>& huffmanCode) {
  if (root == nullptr)
    return;
  if (!root->left && !root->right) {
    huffmanCode[root->ch] = str;
  }
  generateCodes(root->left, str + "0", huffmanCode);
  generateCodes(root->right, str + "1", huffmanCode);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::vector<char> buffer;
  int chunk_size;

  if (rank == 0) {
    std::ifstream input_file("input2.txt", std::ios::ate);
    const auto input_file_size = input_file.tellg();
    input_file.seekg(0, std::ios::beg);
    buffer.resize(input_file_size);
    input_file.read(buffer.data(), input_file_size);
    input_file.close();

    chunk_size = buffer.size() / size;
  }

  // Broadcast chunk size to all processes
  MPI_Bcast(&chunk_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<char> local_buffer(chunk_size);
  MPI_Scatter(buffer.data(), chunk_size, MPI_CHAR, local_buffer.data(), chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

  // Local frequency count
  std::unordered_map<char, int> local_freq;
  for (char c : local_buffer) {
    local_freq[c]++;
  }

  // Send the number of unique characters from each process to root
  std::vector<int> recv_counts(size);
  int local_map_size = local_freq.size();
  MPI_Gather(&local_map_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::unordered_map<char, int> global_freq;

  if (rank == 0) {
    // Root process receives frequency maps and combines them
    for (int i = 0; i < size; ++i) {
      if (i == 0) {
        for (auto& p : local_freq) {
          global_freq[p.first] += p.second;
        }
      } else {
        std::vector<char> keys(recv_counts[i]);
        std::vector<int> values(recv_counts[i]);
        MPI_Recv(keys.data(), recv_counts[i], MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(values.data(), recv_counts[i], MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < recv_counts[i]; ++j) {
          global_freq[keys[j]] += values[j];
        }
      }
    }
  } else {
    // Non-root processes send their frequency maps to root
    std::vector<char> keys;
    std::vector<int> values;
    for (auto& p : local_freq) {
      keys.push_back(p.first);
      values.push_back(p.second);
    }
    MPI_Send(keys.data(), local_map_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    MPI_Send(values.data(), local_map_size, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  if (rank == 0) {
    // Root process builds the Huffman tree
    std::priority_queue<HuffmanTreeNode*, std::vector<HuffmanTreeNode*>, compare> queue;
    for (auto& pair : global_freq) {
      queue.push(new HuffmanTreeNode(pair.first, pair.second));
    }

    while (queue.size() > 1) {
      HuffmanTreeNode* left = queue.top(); queue.pop();
      HuffmanTreeNode* right = queue.top(); queue.pop();
      int sum = left->freq + right->freq;
      queue.push(new HuffmanTreeNode('\0', sum, left, right));
    }

    HuffmanTreeNode* root = queue.top();
    std::unordered_map<char, std::string> codes;
    generateCodes(root, "", codes);

    // Encode the data using generated Huffman codes
    std::string encoded_data;
    for (char c : buffer) {
      encoded_data += codes[c];
    }

    std::vector<char> encoded_data_buffer;
    encoded_data_buffer.push_back(static_cast<char>(encoded_data.length())); // placeholder size

    // Convert binary string to bytes and add to output buffer
    for (size_t i = 0; i < encoded_data.size(); i += 8) {
      std::string byte_string = encoded_data.substr(i, 8);
      if (byte_string.size() < 8)
        byte_string.append(8 - byte_string.size(), '0');
      char byte = static_cast<char>(std::stoi(byte_string, nullptr, 2));
      encoded_data_buffer.push_back(byte);
    }

    // Save encoded binary data to output file
    std::ofstream output_file("output.bin", std::ios::binary);
    output_file.write(encoded_data_buffer.data(), encoded_data_buffer.size());
    output_file.close();

    // Save Huffman codes to file for decoding
    std::ofstream codes_file("codes.txt");
    for (auto& pair : codes) {
      codes_file << pair.first << " " << pair.second << "\n";
    }
    codes_file.close();
  }

  MPI_Finalize();
  return 0;
}
