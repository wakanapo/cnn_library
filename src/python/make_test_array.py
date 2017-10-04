import numpy as np
import pickle

with open('../../test/test_array.hpp', mode='w') as f:
    f.write("#pragma once\n\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Conv1.pkl', 'rb') as fr:
        cnv1 = pickle.load(fr)
        f.write("float conv1_raw[] = {")
        for i, xsss in enumerate(cnv1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Affine1.pkl', 'rb') as fr:
        affine1 = pickle.load(fr)
        f.write("float affine1_raw[] = {")
        for i, xs in enumerate(affine1):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Pool1.pkl', 'rb') as fr:
        pool1 = pickle.load(fr)
        f.write("float pool1_raw[] = {")
        for i, xsss in enumerate(pool1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Relu1.pkl', 'rb') as fr:
        relu1 = pickle.load(fr)
        f.write("float relu1_raw[] = {")
        for i, xsss in enumerate(relu1):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Affine2.pkl', 'rb') as fr:
        affine2 = pickle.load(fr)
        f.write("float affine2_raw[] = {")
        for i, xs in enumerate(affine2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/Relu2.pkl', 'rb') as fr:
        relu2 = pickle.load(fr)
        f.write("float relu2_raw[] = {")
        for i, xs in enumerate(relu2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/x.pkl', 'rb') as fr:
        x = pickle.load(fr)
        f.write("float x_raw[] = {")
        for i, xss in enumerate(x):
            if i != 0:
                f.write(", ")
            for j, xs in enumerate(xss):
                if j != 0:
                    f.write(", ")
                for k, x in enumerate(xs):
                    if k != 0:
                        f.write(", ")
                    f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/matmul1.pkl', 'rb') as fr:
        matmul1 = pickle.load(fr)
        f.write("float matmul1_raw[] = {")
        for i, xs in enumerate(matmul1):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/addbias1.pkl', 'rb') as fr:
        addbias1 = pickle.load(fr)
        f.write("float addbias1_raw[] = {")
        for i, xs in enumerate(addbias1):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/matmul2.pkl', 'rb') as fr:
        matmul2 = pickle.load(fr)
        f.write("float matmul2_raw[] = {")
        for i, xs in enumerate(matmul2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/addbias2.pkl', 'rb') as fr:
        addbias2 = pickle.load(fr)
        f.write("float addbias2_raw[] = {")
        for i, xs in enumerate(addbias2):
            if i != 0:
                f.write(", ")
            for j, x in enumerate(xs):
                if j != 0:
                    f.write(", ")
                f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/conv_dot.pkl', 'rb') as fr:
        cnv_dot = pickle.load(fr)
        f.write("float cnv_dot_raw[] = {")
        for i, xsss in enumerate(cnv_dot):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    with open('/home/wakana/deep-learning-from-scratch-master/ch07/conv_add.pkl', 'rb') as fr:
        cnv_add = pickle.load(fr)
        f.write("float cnv_add_raw[] = {")
        for i, xsss in enumerate(cnv_add):
            if i != 0:
                f.write(", ")
            for j, xss in enumerate(xsss):
                if j != 0:
                    f.write(", ")
                for k, xs in enumerate(xss):
                    if k != 0:
                        f.write(", ")
                    for l, x in enumerate(xs):
                        if l != 0:
                            f.write(", ")
                        f.write(str(x))
        f.write("};\n")
    
