
CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 $(shell pkgconf --cflags sdl3)
LDFLAGS = $(shell pkgconf --libs sdl3)

# Link math library for functions like fminf/fmaxf
LDFLAGS += -lm

SRC = src/main.c src/rasterizer.c
OBJ = $(SRC:.c=.o)

all: render

render: $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f render $(OBJ)

run: render
	./render
