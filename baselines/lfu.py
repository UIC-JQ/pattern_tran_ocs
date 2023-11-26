'''
Author: 娄炯
Date: 2021-01-13 16:31:39
LastEditors: loujiong
LastEditTime: 2021-01-13 22:16:31
Description: lfu implementation based on pqdict
Email:  413012592@qq.com
'''
import pqdict
'''
lfu默认从小到大排序
'''


class LFU:
    '''
    初始化lfu，当中没有任何item
    '''
    def __init__(self):
        self.order = 0
        self.pqdict = pqdict.pqdict()

    '''
    增加某个item的frequency
    self.pqdict[name] 跟字典一样调用和更新
    '''

    def put(self, name):
        if name in self.pqdict:
            self.pqdict[name] = [self.pqdict[name][0] + 1, self.order]
        else:
            self.pqdict[name] = [1, self.order]
        self.order += 1

    '''     
    pop frequency最小的item    item是一个tuple （name，[frequency,内部order]） 内部order是递增的，无意义
    如果为空的话，返回0
    '''

    def pop(self):
        if len(self.pqdict) > 0:
            item = self.pqdict.popitem()
            return item[0]
        else:
            return 0

    '''
    获取某个item的[frequency,内部order]
    如果为空的话，返回0
    '''

    def get(self, name):
        if name in self.pqdict:
            return self.pqdict[name]
        else:
            return 0

    '''
    获取frequency最小的item的name
    如果为空的话，返回0
    '''

    def get_top(self):
        if len(self.pqdict) > 0:
            return self.pqdict.top()
        else:
            return 0

    def get_nsmallest(self,number):
        return pqdict.nsmallest(number,self.pqdict)

    def get_all(self):
        return pqdict.nsmallest(len(self.pqdict), self.pqdict)
    
    def __len__(self):
        return len(self.pqdict)

    def remove(self, name):
        if name in self.pqdict:
            del self.pqdict[name]


def main():
    a = LFU()
    #print(a.__len__())
    a.put("1")
    a.put("33") 
    a.put("33")
    a.put("1")
    a.put("1")
    print(a.get_all())
    a.put("33")
    a.put("2")
    # a.remove("1")
    print(a.get_all())
    #print(a.__len__())
    # print(a.get_nsmallest(10))
    # print(len(a))
    # print(a.get_all())

    # print(a.get_top())
    # print(a.get("10"))
    # print(a.pop())
    # a.put("1")
    # a.put("1")
    # a.put("1")
    # print(a.pop())


if __name__ == "__main__":
    main()
