#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <stdlib.h>

const int c_arity = 3;
const int c_depth = 3;
const int c_length = 27; // arity ^ depth

const bool get_stats = false;

const int c_matrix[c_arity][c_arity] = {
    {1, 0, 0},
    {1, 2, 0},
    {1, 1, 1}
};

const int ref[27][27] = {
    {1, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 2, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 2, 1, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 2, 2, 2, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0, 0, 1, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 1, 2, 0, 1, 2, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0,
     2, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 0, 0, 0, 0, 0, 0,
     2, 1, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 0, 0, 0, 0, 0, 0,
     2, 2, 2, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 2, 0, 0, 0, 0, 0,
     2, 0, 0, 1, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 2, 1, 0, 0, 0, 0,
     2, 1, 0, 2, 2, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 2, 2, 2, 0, 0, 0,
     2, 2, 2, 1, 1, 1, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 1, 0, 0, 1, 0, 0,
     2, 0, 0, 2, 0, 0, 2, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 1, 2, 0, 1, 2, 0,
     2, 1, 0, 2, 1, 0, 2, 1, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1,
     2, 2, 2, 2, 2, 2, 2, 2, 2,
     0, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 0, 0, 0, 0, 0, 0, 0, 0},
    {1, 2, 0, 0, 0, 0, 0, 0, 0,
     1, 2, 0, 0, 0, 0, 0, 0, 0,
     1, 2, 0, 0, 0, 0, 0, 0, 0},
    {1, 1, 1, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 0, 0, 0, 0, 0, 0},
    {1, 0, 0, 2, 0, 0, 0, 0, 0,
     1, 0, 0, 2, 0, 0, 0, 0, 0,
     1, 0, 0, 2, 0, 0, 0, 0, 0},
    {1, 2, 0, 2, 1, 0, 0, 0, 0,
     1, 2, 0, 2, 1, 0, 0, 0, 0,
     1, 2, 0, 2, 1, 0, 0, 0, 0},
    {1, 1, 1, 2, 2, 2, 0, 0, 0,
     1, 1, 1, 2, 2, 2, 0, 0, 0,
     1, 1, 1, 2, 2, 2, 0, 0, 0},
    {1, 0, 0, 1, 0, 0, 1, 0, 0,
     1, 0, 0, 1, 0, 0, 1, 0, 0,
     1, 0, 0, 1, 0, 0, 1, 0, 0},
    {1, 2, 0, 1, 2, 0, 1, 2, 0,
     1, 2, 0, 1, 2, 0, 1, 2, 0,
     1, 2, 0, 1, 2, 0, 1, 2, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1}
};


// If changing value_t, be sure to choose appropriate random initialization
// at the start of the main() function.

//typedef double value_t;
typedef int value_t;

struct workspace {
    int arity;
    int depth;
    int length;
    std::vector<std::vector<int> > matrix;
    std::vector<value_t> in;
    std::vector<value_t> out;
    std::vector<value_t> work;
    std::vector<bool> work_init;
    std::vector<unsigned> index;
    int fullindex;
    std::vector<int> stats_used;
    std::vector<int> stats_from;

    workspace(int a, int d, int l)
        : arity(a),
          depth(d),
          length(l),
          matrix(a),
          in(l),
          out(l, 0),
          work(l),
          work_init(l),
          index(d),
          fullindex(0)
    {
        for (int i = 0; i < a; ++i) {
            std::vector<int> v(&c_matrix[i][0], &c_matrix[i][a]);
            matrix[i].swap(v);
        }

        if (get_stats) {
            stats_used.resize(d);
            stats_from.resize(d);
        }
    }
};

value_t get(workspace& ws, int depth, int base, int len, int offset, int mult)
{
    int index = base + offset + mult - 1;
    if (ws.work_init[index])
        return ws.work[index];
    ws.work_init[index] = true;
    value_t& v = ws.work[index];
    v = 0;

    int matrix_row_index = ws.index[depth];
    const std::vector<int>& row = ws.matrix[matrix_row_index];

    ++depth;
    if (depth == ws.depth) { // base case
        offset *= ws.arity;
        offset /= ws.arity - 1;

        for (int i = 0; i < ws.arity; ++i) {
            int m = row[i];
            if (m != 0) {
                m = (m * mult) % ws.arity;
                v += m * ws.in[offset];
            }
            ++offset;
        }
    } else { // recursive case
        len *= ws.arity;
        base -= len;
        offset *= ws.arity;

        for (int i = 0; i < ws.arity; ++i) {
            int m = row[i];
            if (m != 0) {
                m = (m * mult) % ws.arity;
                v += get(ws, depth, base, len, offset, m);
            }
            offset += ws.arity - 1;
        }
    } // recursive case

    return v;
}

void transform(workspace& ws, int depth, int base, int length, int step)
{
    if (depth == 1) {
        for (int i = 0; i < ws.arity; ++i) {
            ws.index[0] = i;
            ws.work_init[base] = false;
            ws.out[ws.fullindex] = get(ws, 0, base, 2, 0, 1);
            ws.fullindex += step;
        }
        ws.fullindex -= step * ws.arity;
        return;
    }

    std::vector<bool>::iterator wi_b, wi_e;
    wi_b = ws.work_init.begin() + base;
    wi_e = wi_b + length;
    depth -= 1;
    base += length;
    int next_length = length / ws.arity;
    int next_step = step * ws.arity;

    for (int i = 0; i < ws.arity; ++i) {
        std::fill(wi_b, wi_e, false);
        ws.index[depth] = i;
        transform(ws, depth, base, next_length, next_step);

        if (get_stats) {
            ws.stats_used[depth] += std::count(wi_b, wi_e, true);
            ws.stats_from[depth] += length;
        }

        ws.fullindex += step;
    }
    ws.fullindex -= next_step; // (step * ws.arity)
}

void test(const workspace& ws)
{
    for (int i = 0; i < c_length; ++i) {
        value_t v = 0;
        for (int j = 0; j < c_length; ++j) {
            v += ref[i][j] * ws.in[j];
        }
        std::cout << i << " " << v << std::endl;
    }
}

int main()
{
    workspace ws(c_arity, c_depth, c_length);
    srand(1); // seed random manually to have reproducable tests
    for (int i = 0; i < c_length; ++i) {
        
        // value_t double
        //double v = rand();
        //v /= RAND_MAX;

        // value_t int
        int v = rand() % 1024;

        ws.in[i] = v;
    }

    // Only test if c_depth <= 3
    //test(ws);

    transform(ws, c_depth, 0, c_length / c_arity * (c_arity - 1), 1);
    /*
    for (int i = 0; i < c_length; ++i)
        std::cout << i << " " << ws.out[i] << std::endl;
    */

    if (get_stats) {
        unsigned total = 0;
        for (int i = 0; i < c_depth; ++i) {
            total += ws.stats_used[i];
            std::cout << i << ": "
                      << ws.stats_used[i] << "/" << ws.stats_from[i]
                      << std::endl;
        }
        std::cout << total << std::endl;
    }
}
