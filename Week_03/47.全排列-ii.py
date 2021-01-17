#
# @lc app=leetcode.cn id=47 lang=python3
#
# [47] 全排列 II
#

# @lc code=start
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def dsf(nums,size,depth,path,used,res):
            #终止条件
            if depth==size:
                res.append(path[:]) #deep copy
                return
            #当前逻辑
            for i in range(size):
                if not used[i]:
                    if i >0 and nums[i]==nums[i-1] and not used[i-1]:
                        continue
                    used[i]=True    
                    path.append(nums[i])
                    #递归调用
                    dsf(nums,size,depth+1,path,used,res)
                    #状态翻转
                    used[i]=False   #回复元素i的使用状态
                    path.pop()  #撤销元素i

        nums.sort()
        size=len(nums)
        res=[]
        used = [False for i in range(size)]

        #特殊情况返回
        if size==0:
            return res
        
        dsf(nums,size,0,[],used,res)
        return res

# @lc code=end

