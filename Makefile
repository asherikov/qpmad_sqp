BUILD_DIR=build

clean:
	rm -Rf ${BUILD_DIR}

build:
	mkdir -p ${BUILD_DIR}
	cd ${BUILD_DIR}; cmake ..
	cd ${BUILD_DIR}; make

test: build
	cd build; ${MAKE} test

.PHONY: build

