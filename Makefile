# Variables
CXX := arm-linux-gnueabihf-g++
CXXFLAGS := -I /home/user/ncnn/build-arm-linux-gnueabihf/install/include/ncnn \
            -I /opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/include/ \
            -I /usr/local/opencv4.7/install/include/opencv4 \
            -L /usr/local/opencv4.7/install/lib/ \
            -L /home/user/ncnn/build-arm-linux-gnueabihf/install/lib \
            -Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/arm-linux-gnueabihf/libc/lib/ \
            -Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/lib/ \
            -Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/qt5.5_env/lib/ \
            -Wl,-rpath-link=/opt/EmbedSky/gcc-linaro-5.3-2016.02-x86_64_arm-linux-gnueabihf/qt5.5/rootfs_imx6q_V3_qt5.5_env/usr/lib/ \
            -lpthread -lopencv_world -lncnn -std=c++11 -O3
TARGETS := yolox fastest yolov7 yoloXnano yolov4
SRCS := $(addsuffix .cpp, $(TARGETS))
OBJS := $(SRCS:.cpp=.o)

# Règle par défaut
all: $(TARGETS)


# Règle de compilation
$(TARGETS):
	$(CXX) $@.cpp -o $@ $(CXXFLAGS)

# Règle de nettoyage
clean:
	rm -f $(TARGETS) $(OBJS)

# Règle pour l'installation (si nécessaire)
install:
	cp $(TARGETS) /path/to/installation

# Autres dépendances et règles peuvent être ajoutées ici

