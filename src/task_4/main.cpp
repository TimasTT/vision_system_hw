#include <string>
#include <iostream>

#include "frame_matching.h"

const std::string PATH_TO_VIDEO("../data/sample_mpg.avi");

int main() {
    auto using_adaptive_alignment = [](){
        std::cout << "Использовать адаптивное выравнивание?" << std::endl
                  << "1 - Да" << std::endl
                  << "2 - Нет" << std::endl;
        int use = 0;
        std::cin >> use;

        if(use == 1) {
            return true;
        } else if(use == 2) {
            return false;
        } else {
            std::cout << "Выравнивание не используется" << std::endl;
            return false;
        }
    };

    FrameMatching(PATH_TO_VIDEO, using_adaptive_alignment());

    return 0;
}