#include <vector>
#include <iostream>

int main() {
    std::vector<int> v1;
    v1.assign(32, 0);
    std::cout << "size info of v1:" << std::endl;
    std::cout << "size: " << v1.size() << " "
              << "capacity: " << v1.capacity() << std::endl;

    std::vector<int> v2;
    v2.reserve(16);
    v2.assign(8,0);

    std::cout << "size info of v2:" << std::endl;
    std::cout << "size: " << v2.size() << " "
              << "capacity: " << v2.capacity() << std::endl;
    return 0;
}