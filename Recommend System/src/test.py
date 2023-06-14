import pickle
from LFM import FunkSVD
from BiasSVD import BiasSVD
def getResult(model,test_path):
    # 读取test.txt文件
    with open(test_path, 'r') as f:
        lines = f.readlines()

    user = ""
    n_items = ""
    s = ""
    for line in lines:
        if "|" in line:
            with open('result.txt', 'a') as f:
                f.write(s)
            s = line
            user, n_items = line.strip().split("|")
        else:
            item = line.strip()
            # item = int(item)
            score = model._predict(user, item)
            s += str(item)
            s += " "
            s += str(score)
            s += "\n"
    with open('result.txt', 'a') as f:
        f.write(s)

if __name__=='__main__':
    m_path=input('请输入模型路径:')
    print('模型载入中，请耐心等候...')
    with open(m_path,'rb') as f:
        model=pickle.load(f)
    print('模型载入成功')
    test_path=input('请输入测试文件路径:')
    getResult(model,test_path)