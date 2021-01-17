#
# @lc app=leetcode.cn id=77 lang=python3
#
# [77] 组合
#

# @lc code=start
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        #生成数组
        nums = [i for i in range(1,n+1)]
        # print(nums)
        res=[]

        def backtrace(cur_res,idx):
            # print("cur_res:",cur_res)
            if len(cur_res)==k:
                res.append(cur_res[:]) #浅拷贝
                return
            
            for i in range(idx,n+1):
                cur_res.append(i)
                backtrace(cur_res,i+1)
                cur_res.pop()
            
        if n==0 or k==0:
            return res
        
        backtrace([],1)
        return res
        


# @lc code=end

