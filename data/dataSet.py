import shutil
import sys
import os


def walkdir(file):
    for root, dirs, files in os.walk(file):
        for f in files:
            m = os.path.join(root, f)
            print("m=", m)
            ss = os.path.splitext(m)
            print("ss = ", ss)
            dirname = ss[0]  # 文件夹
            print("dirname = ", dirname)
            a = os.path.basename(m)  # 文件名，带后缀
            file_name = a.split('.')[0]
            houzui = a.split('.')[0]
            print("file_name = ", file_name, "--------houzui = ", houzui)
            print("a=", a)

            print('root=', root.split('\\')[-1])
            print(root + '\\' + root.split('\\')[-1] + '-' + houzui + '.jpg')

            name2 = root + '\\' + root.split('\\')[-1] + '-' + houzui + '.jpg'

            os.rename(m, name2)


# 下面一句的作用是：运行本程序文件时执行什么操作


def mymovefile(srcfile, dstpath):  # 移动函数

    shutil.copy(srcfile, dstpath)  # 移动文件


if __name__ == "__main__":
    path = 'D:\WorkProject\DataSet\Weather\weather\dataset'
    train_path = 'train_data/'
    test_path = 'test_data/'
    for root, dirs, files in os.walk(path):
        if len(files) != 0:
            print(len(files))
            i = 1
            for f in files:
                if i < (len(files) * 0.9):
                    mymovefile(root + '\\' + f, train_path + f)
                else:
                    mymovefile(root + '\\' + f, test_path + f)
                i = i + 1
