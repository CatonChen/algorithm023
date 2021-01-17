#
# @lc app=leetcode.cn id=70 lang=python3
#
# [70] 爬楼梯
#

# 整数
#1 1 
#2 1+1 or 2
#3 1+1+1 or 2+1 or 1+2



# @lc code=start
class Solution:
    def climbStairs(self, n: int) -> int:
        #处理n<=2时
        if n ==1 or n ==2:
            return n
        #初始化变量
        a , b , tmp = 1, 2 ,0
        for i in range(3,n+1):
            tmp=a+b #f(n) = f(n-1) + f(n-2)
            a=b
            b=tmp
        return tmp


# @lc code=end

