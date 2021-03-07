# WEEK09学习笔记
### 基础知识
#### 递归
1. 函数自身调用自身称之为递归。
	- 递——逐层往下调用自身
	- 归——逐层往上返回结果
2. 递归代码模板
```python
# Pythondef recursion(level, param1, param2, ...):     
	# recursion terminator    
	#递归终止条件，没有这个会陷入死循环 
	if level > MAX_LEVEL: 	   
		process_result 	   
		return     
	# process logic in current level  
	#解决业务逻辑代码   
	process(level, data...)     
	# drill down    
	#递归调用自身 
	self.recursion(level + 1, p1, ...)     
	# reverse the current level status if needed
	#根据需要清理全局变量或其他因素状态
```
3. 思维
	1. 最大误区，人肉递归
	2. 找最近最简有效性方法，将原问题拆解至重复子问题求解。
	3. 数学归纳法的思维。

#### 分治
1. 分治的本质是递归，仍然是找解决问题的重复性。
2. 分治具体过程表现是将原始问题拆解成若干个子问题，分别对子问题求解，最后将子问题解合并成原始问题的解。
3. 分治代码模板
```python
def divide_conquer(problem, param1, param2, ...): 
	# recursion terminator
	if problem is None:
	    print_result
	return
  	# prepare data
	data = prepare_data(problem)
	subproblems = split_problem(problem, data)
	# conquer subproblems
	subresult1 = self.divide_conquer(subproblems[0], p1, ...)
	subresult2 = self.divide_conquer(subproblems[1], p1, ...)
	subresult3 = self.divide_conquer(subproblems[2], p1, ...)
	#...
	# process and generate the final result
	result = process_result(subresult1, subresult2, subresult3, ...)
  	# revert the current level states
```
5. 分治+最优子结构=动态规划

#### 动态规划
1. 关键点
	1. 动态规划适用于具有**重叠子问题**和**最优子结构**性质的问题。
	2. 动态规划具有剪枝效果（**淘汰次优解**），因其对子问题仅求一次最优解；将子问题的解记忆化存储，遇到重复子问题时直接查表，降低时间复杂度。
2. dp代码模板
```java
function DP():
	dp = [][] # 二维情况
	for i = 0 .. M { for j = 0 .. N {
		dp[i][j] = _Function(dp[i’][j’]...) }
  	}
  	return dp[M][N];
```
3. 常见问题
	1. 爬楼梯
	2. 不同路径
	3. 打家劫舍
	4. 最小路径和
	5. 买卖股票
4. 不同路径2动态规划算法思路
	1. 在62.不同路径的基础上增加了障碍物的设定。
	2. 因为是从左上角走到右下角，且每次只能右移一步或下移一步，故可以得到状态转移方程：dp[i][j]=dp[i-1][j]+dp[i][j-1]。即走到dp[i][j]的不同路径和等于它上面dp[i-1][j]的路径和加上它左面dp[i][j-1]的路径和。
	3. 当因为障碍物的出现，故当obstacleGrid[i][j]=1时，相应的dp[i][j]=0。
	4. 所以整个dp方程为以下：
		1. 当obstacleGrid[i][j]=1时，dp[i][j]=0
		2. 当obstacleGrid[i][j]!=1时，dp[i][j]=dp[i-1][j]+dp[i][j-1]
	5. dp数组初始化时仍需将上边界和左边界初始为1，但要注意障碍物的出现：
		1. 当obstacleGrid[0][j]=1时，只有[0,j)初始为1，[j,n)保持0
		2. 同理当obstacleGrid[i][0]=1时，只有[0:i)初始为1，[i:m)保持0不变。
	6. 最终返回dp[-1][-1]的值即可。

#### 高级动态规划
1. dp状态拥有更多维度（二维、三维或者更多，空间考虑状态压缩）
2. 状态方程更加复杂
3. 常见问题
	1. 使用最小花费爬楼梯
	2. 编辑距离

#### 字符串
- python、java等语言的字符串是不可变的。
- c和c\+\+的字符串因为指针的缘故，所以是可变的。
- 常见的字符串应用操作：遍历、比较、查询和替换
**常见字符串问题场景**
1. 字符串操作问题
2. 字符串异位词问题
3. 字符串回文串问题
**高级字符串算法**
1. 最长子串、子序列问题
2. 字符串+动态规划
3. 字符串匹配算法
	1. 暴力法
	2. rabin-karp算法
	3. kmp算法

### 本周leetcode练习总结
#### 387.字符串中的第一个唯一字符
1. 哈希字典
	1. 算法思路
		1. 创建一个哈希字典表hashmap，遍历一次字符串s，在字典中记录字符串s中每个字符及其出现的次数。
		2. 再次遍历一次字符串s，返回第一个其次数等于1的元素的下标i。
	2. 复杂度分析
		1. 时间复杂度：O(N)，虽然遍历两次字符串s，但并非嵌套循环。
		2. 空间复杂度：O(N)，哈希字典利用了额外空间。
	3. 题解代码
```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        # 哈希字典记录字符和次数
        hashmap = {}
        for c in s:
            if hashmap.get(c) is not None:
                hashmap[c] += 1
            else:
                hashmap[c] = 1
        # print(hashmap)
        # 遍历字符串，返回第一个次数为1的元素下标i
        for i in range(len(s)):
            if hashmap[s[i]] == 1:
                return i
        return -1
```

#### 51.N皇后（本周复习）
1. 回溯
	1. 算法思路
		1. 根据N皇后的定义，即**每一行有且仅有一个皇后，每一列有且仅有一个皇后，及每一条主对角线和副对角线上有且仅有一个皇后**。
		2. 递归&剪枝过程
			1. 递归终止条件：当行row==n时，即表示找到一个可行解，将可行解cur\_res加入结果res中。
			2. 预先定义列cols、撇pie和捺na的集合set，用于保存已经访问过的列及主副对角线。
			3. 当固定行row时，遍历每一列col，当col存在于cols，或row-col存在于pie，或row+col存在于na时，表示相应的行、主对角线和副对角线已经有皇后Q的存在了，需要进行剪枝操作。
			4. 否则，将col加入cols、row-col加入pie和row+col加入na；继续调用自身函数进行递归工作，传入的row+1，且中间结果cur\_res+[col]，表示该col可以放置一个Q。
			5. 递归后，进行状态撤销以便回溯。
		3. 当所有N\*N的单元格都遍历完毕后，根据结果res保存的皇后位置的下标，画出相应的棋盘状态。
	2. 复杂度分析
		1. 时间复杂度：O(N!)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    # 定义回溯
    def dfs(self, n, row, cur_res):
        # 当行==n时，说明找到可行解，记录中间结果。
        if row == n:
            self.res.append(cur_res)
            return
        # 遍历每列
        for col in range(n):
            # 剪枝
            if col in self.cols or row + col in self.pie or row - col in self.na:
                continue
            self.cols.add(col)
            self.pie.add(row + col)
            self.na.add(row - col)
            # 递归工作，行+1，占用的列加入结果
            self.dfs(n, row + 1, cur_res + [col])
            # 撤销
            self.cols.remove(col)
            self.pie.remove(row + col)
            self.na.remove(row - col)

    def solveNQueens(self, n: int) -> List[List[str]]:
        self.res = []
        if n < 1:
            return res
        # 列，撇，捺
        self.cols = set()
        self.pie = set()
        self.na = set()
        self.dfs(n, 0, [])
        return [['.'*i+'Q'+'.'*(n-i-1) for i in sol] for sol in self.res]
```
4. 国际版高赞题解参考
```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # 回溯
        def dfs(queens, xy_diff, xy_sum):
            p = len(queens)  # 算行数，行列相等
            if p == n:
                res.append(queens)
                return
            # 遍历每列
            for q in range(n):
                # 剪枝 ，逆向思维，能放Q的位置必须同时满足不在列、撇、捺里
                if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                    # 递归操作
                    dfs(queens + [q], xy_diff + [p - q], xy_sum + [p + q])

        res = []
        dfs([], [], [])
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in sol] for sol in res]
```

#### 303.区域和检索 - 数组不可变
1. 前缀和
	1. 算法思路
		1. 利用前缀和方法presum快速计算指定区间i-j的元素之和。
		2. 定义一个n+1的presum数组，presum[i]表示该元素左边的所有元素之和（不包含i元素）。遍历一次数组，累加区间[0,i)范围内的元素，得到presum数组，即：**presum[i+1]=presum[i]+nums[i]**。
		3. 利用presum数组，求nums数组任意区间[i,j]之间的元素和，可以通过**sum(i,j)=presum[j+1]-presum[i]**得到。
	2. 复杂度分析
		1. 时间复杂度：O(N)，构造presum数组需要O(N)
		2. 空间复杂度：O(N)，presum数组所需空间
	3. 题解代码
```python
class NumArray:

    def __init__(self, nums: List[int]):
        n = len(nums)
        self.presum = [0] * (n + 1)
        for i in range(n):
            self.presum[i + 1] = self.presum[i] + nums[i]

    def sumRange(self, i: int, j: int) -> int:
        return self.presum[j + 1] - self.presum[i]
```

#### 62.不同路径（本周复习）
1. 动态规划
	1. 算法思路
		1. 规划一个dp二维数组dp[i][j]，代表m\*n的网格。因为从左上角到右下角，只能向右走或向下走，所以到达右下角的路径和必然是到达其前面相邻的两个格子的路径和。故得到状态转移方程：**dp[i][j]=dp[i-1][j]+dp[i][j-1]**。
		2. 其中需要注意的点：
			1. 初始dp[0][0]=1，表示网格起点为1。
			2. 初始dp[i][0]和dp[0][j]=1，表示网格两个边界的值均为1，因为这些格子只能从其前面相邻的格子走过来。
				![]()
		3. 最终返回dp[-1][-1]即可。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，遍历m\*n的网格的所有单元格。
		2. 空间复杂度：O(M\*N)，dp数组状态记录所需
	3. 题解代码
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 构建二维dp数组
        dp = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        # print(dp)
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # print(dp)
        return dp[-1][-1]
```
4. 状态空间优化
	1. 将dp数组从二维优化成一维数组，因为当前dp[i][j]的值，只与dp[i-1][j]和dp[i][j-1]有关，即和左边和上边的网格有关。
		2. 故状态转移方程可优化为：**dp[j]=dp[j]+dp[j-1]**。
		3. 复杂度分析
			1. 时间复杂度：O(M\*N)
			2. 空间复杂度：O(N)
		4. 题解代码
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # 一维数组dp
        dp = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                dp[j] += dp[j - 1]
        # print(dp)
        return dp[-1]
```

#### 53.最大子序和（本周复习）
1. 动态规划
	1. 算法思路
		1. 定义一维数组dp记录状态，dp长度为数组nums长度，初始dp[0]=nums[0]。
		2. 假设dp[i]为最大子序和，那么dp[i]有两种情况：
			1. 从当前子数组起点x连续到i-1的子序和，再加上nums[i]本身之后的子序和为当前最大子序和。
			2. nums[i]本身即为当前最大子序和。
		3. 故得到状态转移方程：**dp[i]=max(dp[i-1]+nums[i],nums[i])**。最终返回dp数组中的最大值。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，dp数组空间所需
	3. 题解代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp数组
        dp = [0] * len(nums)
        dp[0] = nums[0]
        # 遍历一次数组
        for i in range(1, len(nums)):
            dp[i] = max(dp[i - 1] + nums[i], nums[i])
        return max(dp)
```

2. 贪心算法
	1. 算法思路
		1. 局部最优：当前连续和是负数时，就放弃，并从下一位开始重新计算连续和。
		2. 全局最优：从所有局部最优中找出最大者。在计算局部最优的情况下，记录其中最大者即为全局最优。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 贪心算法
        res = float('-inf')  # 设定res初始为最小值
        tmp = 0
        for i in range(len(nums)):
            tmp += nums[i]  # 当前子数组连续和
            if tmp > res:
                res = tmp  # tmp>res，则记录最大的tmp
            if tmp < 0:  # 当前子数组连续和<0时，tmp重置为0
                tmp = 0
        return res
```

#### 151.翻转字符串里的单词
1. 利用库函数
	1. 算法思路
		1. 利用语言提供的库函数对字符串操作，达成目的。
		2. reversed、split、strip和join
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(s.strip().split()[::-1])
```
4. 用reversed替代[:\:-1]
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(s.strip().split()))
```

2. 双指针
	1. 算法思路
		1. 倒序遍历字符串s，用双指针i,j记录单词的左右边界下标。
		2. 每确定一个单词，将其添加到单词列表res中。
		3. 最后将单词列表拼接成字符串返回即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        # 删除首尾空格
        s = s.strip()
        i = j = len(s) - 1  # 初始化i,j下标
        res = []
        # 遍历字符串
        while i >= 0:
            # 搜索走出单词的第一个空格
            while i >= 0 and s[i] != ' ':
                i -= 1
            # 添加单词
            res.append(s[i + 1:j + 1])
            # 跳过单词之间的空格
            while i >= 0 and s[i] == ' ':
                i -= 1
            # 移动j到i处
            j = i
        # 返回结果
        return ' '.join(res)
```

541.反转字符串II
1. 暴力
	1. 算法思路
		1. 将字符串s转换为数组a。
		2. 遍历数组，下标移动每次加2\*k，同时利用reversed函数翻装a[i:i+k]范围内的元素位置，实现翻转目的。
		3. 最后将数组a字符串化后返回。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，将字符串s数组化
	3. 题解代码
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        # 数组化s
        nums = list(s)
        for i in range(0, len(nums), 2 * k):    #下标每次移动2k间隔
            nums[i:i + k] = reversed(nums[i:i + k])
        return ''.join(nums)
```
4. 利用切片的题解代码。因为没有使用额外数组，所以空间复杂度为O(1)。
```python
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        res = ''
        for i in range(0, len(s), 2 * k):
            tmp = s[i:i + k]
            tmp = tmp[::-1] + s[i + k:i + 2 * k]
            res += tmp
        return res
```

#### 55.跳跃游戏
1. 贪心算法
	1. 算法思路
		1. 依次遍历数组nums，同时维护一个最大可跳距离i+nums[i]。
		2. 当到达最后一个元素的位置时，如果最大可跳距离可以覆盖这个距离，那么可以返回true，否则返回false。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历数组
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n, tmp = len(nums), 0
        for i in range(n):
            if i <= tmp:
                tmp = max(tmp, i + nums[i])
                # print(tmp)
                if tmp >= n - 1:
                    return True
        return False
```

#### 300.最长递增子序列
1. 动态规划
	1. 算法思路
		1. dp[i]表示，以nums[i]为结尾的上升子序列的长度。这个定义中，nums[i]必须被选取，且必须是子序列的最后一个元素。
		2. 如果一个较大的数在较小的数后面，会形成一个更长的子序列。只要nums[i]严格大于在它之前的某个数，那么nums[i]就可以接在这个数后面形成一个更长的上升子序列。故得到状态转移方程：**dp[i]=max(dp[i],dp[j]+1)**。
		3. 初始化时dp数组均为1。一个字符是长度为1的上升子序列。
		4. 最终返回整个dp状态数组中的最大值。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)，因为遍历两次数组，两次皆是线性循环。
		2. 空间复杂度：O(N)，dp数组利用空间等于输入数组空间大小。
	3. 题解代码
```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        # 初始化dp
        dp = [1] * n
        # 循环数组,i从1开始。
        for i in range(1, n):
            # 0<=j<i
            for j in range(i):
                # nums[i]>nums[j]时，更新长度
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        # print(dp)
        return max(dp)
```

#### 557.反转字符串中的单词III
1. 利用队列与栈
	1. 算法思路
		1. 队列是先进先出；栈是先进后出。
		2. 遍历字符串，有遇到单词和空格两种情况：
			1. 遇到单词时，将字符逐个压入栈。
			2. 遇到空格时，将栈里的元素逐个出栈加入队列，即完成单个单词的反转。末了，将空格加入队列。
		3. 遍历完字符串后，记得将栈内最后的单词加入队列。
		4. 最终将队列转为字符串返回即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次字符串，和最后遍历一次栈，但两者不是嵌套循环。
		2. 空间复杂度：O(N)，队列和栈都利用了额外空间。
	3. 题解代码
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        queue = []
        stack = []
        for c in s:
            if c==' ':
                while stack:
                    queue.append(stack.pop())
                queue.append(c)
            else:
                stack.append(c)
        while stack:
            queue.append(stack.pop())
        return ''.join(queue)
```

2. 切片
	1. 算法思路
		1. 先通过split函数将字符串拆分成单词列表。
		2. 遍历单词列表，利用切片[:\:-1]反转单词；最终返回数组的字符串即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(word[::-1] for word in s.split(' '))
```

#### 15.三数之和（本周复习）
1. 双指针
	1. 算法思路
		1. 先将数组nums排序。
		2. 先固定一个元素k，然后i和j分别代表数组nums[k+1:n-1]的左右边界，找nums[k]+nums[i]+nums[j]=0的可行解。
		3. 关于元素k有以下两点要考虑：
			1. 如果nums[k]\>0，则必然不会有nums[k]+nums[i]+nums[j]=0的解。
			2. 当k\>0时，如果nums[k]==nums[k-1]，则跳过重复的元素k。
		4. 当i\<=j循环时，令s=nums[k]+nums[i]+nums[j]，有以下几个场景要判断：
			1. 当s==0时，表示找到一组解，将[nums[k],nums[i],nums[j]]添加进结果，同时i和j各自移动一步。
			2. 当s\>0时，表示和值太大，需要移动j。
			3. 当s\<0时，表示和值太小，需要移动i。
			4. 上述1-3点，在移动i和j时，需要跳过i和j的重复元素，避免重复解。
	2. 复杂度分析：
		1. 时间复杂度：O(N^2)，元素k是外层循环，i和j是内层循环，是两层嵌套循环。
		2. 空间复杂度：O(1)，常数变量。但保存结果的res数组是O(N)。
	3. 题解代码
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()  # 排序
        n = len(nums)
        if n < 3:  # 少于3个数，没有解
            return res
        for k in range(n - 2):
            if nums[k] > 0:  # 当nums[k]>0时，没有解。
                break
            if k > 0 and nums[k] == nums[k - 1]:  # 跳过重复元素
                continue
            i, j = k + 1, n - 1
            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s == 0:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                elif s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                else:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
        return res
```

2. 递归法
	1. 算法思路
		1. 递归法是求n数之和的思路。主要是将n数之和逐步分解到2数之和。主要递归工作为：
			1. 当n\>=3时，固定第一个元素k后，将后面的元素和target-k作为新的数组和目标值传给自身函数进行递归工作。
			2. 当n==2时，通过双指针法在数组中寻找nums[i]+nums[j]=target的可行解。
			3. a.递归参数的数组长度\<n时返回空解；b.注意跳过重复元素。
		2. 初始需要对nums数组进行排序。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)，递归调用的系统栈。
	3. 题解代码
```python
class Solution:
    def nSum(self, nums, n, target):
        res = []
        if len(nums) < n:
            return res
        # n==2
        if n == 2:
            i, j = 0, len(nums) - 1
            while i < j:
                s = nums[i] + nums[j]
                if s == target:
                    res.append([nums[i], nums[j]])
                    while i < j and nums[i] == nums[i + 1]:
                        i += 1
                    while i < j and nums[j] == nums[j - 1]:
                        j -= 1
                    i += 1
                    j -= 1
                elif s < target:
                    i += 1
                else:
                    j -= 1
            return res
        else:  # n>2
            for k in range(len(nums)):
                if k > 0 and nums[k] == nums[k - 1]:
                    continue
                else:
                    subres = self.nSum(nums[k + 1:], n - 1, target - nums[k])
                    for p in range(len(subres)):
                        res.append([nums[k]] + subres[p])
            return res

    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        return self.nSum(nums, 3, 0)
```

#### 64.最小路径和（本周复习）
1. 动态规划
	1. 算法思路
		1. 因为只能从左上角走到右下角，且只能右移一步或下移一步，所以m\*n的矩阵grid对应的二维dp数组有以下四种可能：
			1. 当i和j都不是边界时，即i!=0且j!=0，此时的dp[i][j]=min(dp[i-1][j],dp[i][j-1])+grid[i][j]。
			2. 当i=0时，即沿着上边界移动，则有dp[i][j]=dp[i][j-1]+grid[i][j]。
			3. 同理j=0时，即沿着左边界移动，则有dp[i][j]=dp[i-1][j]+grid[i][j]。
			4. 初始dp[0][0]=grid[0][0]。
		2. 最终返回dp[-1][-1]即可。
		3. 如果在原矩阵上做修改，则无需额外空间记录状态。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，遍历矩阵m\*n。
		2. 空间复杂度：O(1)，在原矩阵上修改。
	3. 题解代码
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == 0 and j == 0:
                    grid[0][0] = grid[0][0]
                elif i == 0:
                    grid[i][j] = grid[i][j - 1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i - 1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]
```

#### 232.用栈实现队列
1. 利用栈
	1. 算法思路
		1. 根据栈的先进后出的特点，利用两个栈实现队列的先进先出。
		2. push方法：加元素压入第一个栈s1即可。
		3. pop方法：当第二个栈s2为空时，将s1里的元素全部弹出，压入s2，最后弹出s2的栈顶元素即可。
		4. peek方法：同第3点，但不弹出s2的栈顶元素，只需返回它s2[-1]。
		5. empty方法：当s1或s2的长度都不为0时，即不为空false，否则true。
	2. 复杂度分析
		1. 时间复杂度：push是O(1)；pop/peek平均是O(1)，最差O(N)。
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        # 两个栈
        self.s1 = []
        self.s2 = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.s1.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()

    def peek(self) -> int:
        """
        Get the front element.
        """
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2[-1]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        if len(self.s1) != 0 or len(self.s2) != 0:
            return False
        else:
            return True
```

#### 917.仅仅反转字母
1. 利用栈
	1. 算法思路
		1. 利用栈先进后出的特性完成字符串的反转。
		2. 遍历一次字符串，将字符串中的所有字母入栈。
		3. 再遍历一次字符串，当遇到字母时，弹出栈顶元素加入结果；当遇到非字母时，直接将字符加入结果。
		4. 最终返回字符串化的结果。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def reverseOnlyLetters(self, S: str) -> str:
        letters = [c for c in S if c.isalpha()]
        # print(letters)
        res = []
        for c in S:
            if c.isalpha():
                res.append(letters.pop())
            else:
                res.append(c)
        return ''.join(res)
```

#### 205.同构字符串
1. 双哈希表
	1. 算法思路
		1. 如果是同构字符串，则有s中的任意一个字符被t中唯一的字符对应，反之亦然。这个现象称为“双射”关系。
		2. 建立两个哈希表s2t和t2s，遍历一次s的长度，同时将s2t[s[i]]=t[i]的关系记录下来，同理可得t2s[t[i]]=s[i]。
		3. 当s[i]已经在s2t中，且s[i]!=t[i]时，即s和t不能成为同构字符串。反之亦然。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)，最差O(N)。
	3. 题解代码
```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        s2t = {}
        t2s = {}
        for i in range(len(s)):
            x, y = s[i], t[i]
            if (x in s2t and s2t[x] != y) or (y in t2s and t2s[y] != x):
                return False
            s2t[x] = y
            t2s[y] = x
        # print(s2t)
        # print(t2s)
        return True
```

2. 利用index函数
	1. 算法思路
		1. 根据index函数返回元素的下标来判断两边是否相等的结果
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        for i in range(len(s)):
            # print(s.index(s[i]))
            # print(t.index(t[i]))
            if s.index(s[i]) != t.index(t[i]):
                return False
        return True
```

#### 680.验证回文字符串II
1. 贪心算法
	1. 算法思路
		1. 回文字符串的特点是头尾两个字符相等，所以可以定义下标i和j指向字符串的头尾两个字符。
		2. 判断回文字符串的方法是：**当i\<j的时候，判断元素i和元素j是否相等，相等继续移动i和j，直到i\>=j迭代完毕，返回true；否则不相等时返回false**。
		3. 因为可以删除其中一个字符串，所以在移动下标i和j的时候，当碰到当前元素i和元素j不相等时，那么可以删除其中一个，并继续判断剩余字符串是否为回文串。
		4. **剩余字符串为s[i+1:j]或s[i:j-1]**；调用第2点的方法判断剩余字符串是否为回文字符串，当其中一个为回文串时，说明原始字符串是回文字符串。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次字符串s
		2. 空间复杂度：O(1)，常数变量
	3. 题解代码
```python
class Solution:
    def validPalindrome(self, s: str) -> bool:
        # 定义check判断是否回文字符串
        def check(i, j):
            while i < j:
                if s[i] != s[j]:
                    return False
                else:
                    i += 1
                    j -= 1
            return True

        # 判断s是否为回文
        i, j = 0, len(s) - 1
        while i < j:
            if s[i] == s[j]:
                i += 1
                j -= 1
            else:
                return check(i + 1, j) or check(i, j - 1)
        # 没有不同的i,j，返回true
        return True
```

#### 70.爬楼梯（本周复习）
1. 递归
	1. 算法思路
		1. 根据f(n)=f(n-1)+f(n-2)的公式，可以通过调用函数自身来进行递归工作，最终得到n的时候的返回值。
		2. 递归终止：n\<=2时，返回n即可。
		3. 因为递归涉及大量重复计算，所以需要lru\_cache进行记忆化搜索。
	2. 复杂度分析
		1. 时间复杂度：O(N)，如果没有记忆化结果，可能是O(N!)
		2. 空间复杂度：O(N)，lru\_cache利用的空间
	3. 题解代码
```python
class Solution:
    @functools.lru_cache(1000)
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        return self.climbStairs(n - 1) + self.climbStairs(n - 2)
```

2. 动态规划
	1. 算法思路
		1. 状态转移方程：dp[i]=dp[i-1]+dp[i-2]。
		2. n\<=2时，返回n即可。
		3. 可以不用创建dp数组记录每一步dp[i]的结果。可以通过变量tmp记录当前dp[i-1]+dp[i-2]的结果用于下次计算，达到节约空间的目的。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n <= 2:
            return n
        a, b, tmp = 1, 2, 0
        for i in range(3, n + 1):
            tmp = a + b
            a = b
            b = tmp
        return tmp
```

3. 斐波那契数列的计算公式
	1. 算法思路
		1. 记住数学通项公式
	2. 复杂度分析
		1. 时间复杂度：O(1)，不计算库函数的时间复杂度
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        import math
        sqrt5 = 5 ** 0.5
        fib = math.pow((1 + sqrt5) / 2, n + 1) - math.pow((1 - sqrt5) / 2, n + 1)
        return int(fib / sqrt5)
```

#### 63.不同路径II
1. 动态规划
	1. 算法思路
		1. 在62.不同路径的基础上增加了障碍物的设定。
		2. 因为是从左上角走到右下角，且每次只能右移一步或下移一步，故可以得到状态转移方程：dp[i][j]=dp[i-1][j]+dp[i][j-1]。即走到dp[i][j]的不同路径和等于它上面dp[i-1][j]的路径和加上它左面dp[i][j-1]的路径和。
		3. 当因为障碍物的出现，故当obstacleGrid[i][j]=1时，相应的dp[i][j]=0。
		4. 所以整个dp方程为以下：
			1. 当obstacleGrid[i][j]=1时，dp[i][j]=0
			2. 当obstacleGrid[i][j]!=1时，dp[i][j]=dp[i-1][j]+dp[i][j-1]
		5. dp数组初始化时仍需将上边界和左边界初始为1，但要注意障碍物的出现：
			1. 当obstacleGrid[0][j]=1时，只有[0,j)初始为1，[j,n)保持0
			2. 同理当obstacleGrid[i][0]=1时，只有[0:i)初始为1，[i:m)保持0不变。
		6. 最终返回dp[-1][-1]的值即可。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，M和N分别为矩阵的长宽。
		2. 空间复杂度：O(M\*N)，dp矩阵
	3. 题解代码
```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        # 初始化dp数组
        dp = [[0] * n for _ in range(m)]
        # 初始化dp上边界和左边界,遇到障碍物之前的都为1
        for i in range(m):
            if obstacleGrid[i][0] != 1:
                dp[i][0] = 1
            else:
                break
        for j in range(n):
            if obstacleGrid[0][j] != 1:
                dp[0][j] = 1
            else:
                break
        # 遍历矩阵
        for i in range(1, m):
            for j in range(1, n):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        # print(dp)
        return dp[-1][-1]
```

#### 72.编辑距离
1. 动态规划
	1. 算法思路
		1. 对于单词A和B，可以对任意一个单词进行插入、删除和替换的操作。进行剖析后，可以得知：
			1. 对A进行插入一个字母等同于对B删除一个字母；反之亦然。
			2. 对A替换一个字母等同于对B替换一个字母。
		2. 因此通过动态规划解决这个问题，对于二维数组dp来说有以下几种情况考虑：
			1. 当A[i]=B[j]时，dp[i][j]=dp[i-1][j-1]的编辑次数
			2. 当A[i]!=B[j]时，最小编辑次数dp[i][j]有以下三种情况考虑：
				1. dp[i][j]=dp[i-1][j]+1
				2. dp[i][j]=dp[i][j-1]+1
				3. dp[i][j]=dp[i-1][j-1]+1
		3. 当对单词A和B，其长度分别为m和n，遍历完毕后，最终最小编辑次数为dp[m][n]。
		4. 初始dp数组时，dp矩阵的两个边界需要初始为0-单词长度，如\0-m和0-n。
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，遍历单词A\*单词B的矩阵
		2. 空间复杂度：O(M\*N)，dp数组所需
	3. 题解代码
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        # 初始化dp
        m, n = len(word1), len(word2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        # print(dp)
        # 遍历word1*word2的矩阵，记录状态
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:  # 字母相等
                    dp[i][j] = dp[i - 1][j - 1]
                else:  # 字母不等
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        # print(dp)
        # 返回dp中m,n的位置
        return dp[m][n]
```

#### 32.最长有效括号
1. 动态规划
	1. 算法思路
		1. 定义dp[i]表示下标i字符结尾的最长有效括号的长度。因为有效子串肯定以“）”结尾，那么子串的开始的“（”的dp值肯定为0，求“）”在dp数组中的位置值。dp数组初始为0。
		2. 遍历字符串s，求解dp值，因为括号成对，所以每两个字符检查一次，有几下情况考虑：
			1. 当s[i]=‘)’ 且 s[i-1]=‘(’时，说明相邻的左右括号成对，推出公式：dp[i]=dp[i-2]+2。
			2. 当s[i]=‘)’且s[i-1]=‘)’时，说明i下标的右括号需要和更前面的左括号匹配成对，如果有s[i-dp[i-1]-1]=‘(’，则推出公式：dp[i]=dp[i-1]+dp[i-dp[i-1]-2]+2。
		3. 最后返回dp数组中的最大值即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次字符串s。
		2. 空间复杂度：O(N)，dp数组所需字符串长度的一维空间。
	3. 题解代码
```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        if n == 0:
            return 0
        # dp数组
        dp = [0] * n
        for i in range(1, n):  # 从下标1开始，()最短
            if s[i] == ')':
                if s[i - 1] == '(':
                    dp[i] = dp[i - 2] + 2
                # i - dp[i - 1] - 1 >= 0 保证下标有效
                elif s[i - 1] == ')' and i - dp[i - 1] - 1 >= 0 and s[i - dp[i - 1] - 1] == '(':
                    dp[i] = dp[i - 1] + dp[i - dp[i - 1] - 2] + 2
        # print(dp)
        return max(dp)
```

#### 74.搜索二维矩阵（本周复习）
1. 线性扫描+二分查找
	1. 算法思路
		1. 因矩阵的特点：a.每行都是生序，b.下一行的首部元素\>上一行的末尾元素。那么可以先通过target与矩阵每行的末尾元素比较，确定target在哪行，后续只要对该行进行查找即可。若target\>矩阵每行的末尾元素，则说明target不在矩阵中。
		2. 在第1点中确认了target的行数后，即可对该行进行二分查找确认target的位置。
			1. 设left=0和right=len(matrix[index])-1，则有mid=(left+right)/2。
			2. 当matrix[index][mid]==target时，返回true。
			3. 当matrix[index][mid]\<target时，移动left=mid+1；否则移动right=mid-1即可。
			4. 如果没有找到则返回false。
	2. 复杂度分析
		1. 时间复杂度：O(logN+M)，二分查找的时间复杂度为O(logN)，M为遍历矩阵行数。
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix:
            return False
        index = -1
        # 遍历matrix每行
        for i in range(len(matrix)):
            if target <= matrix[i][-1]:
                index = i
                break
        # 根据index结果，确认target是否在matrix中
        if index == -1:
            return False
        else:  # 二分查找
            left, right = 0, len(matrix[index]) - 1
            while left <= right:
                mid = (left + right) >> 1
                if target == matrix[index][mid]:
                    return True
                elif target > matrix[index][mid]:
                    left = mid + 1
                else:
                    right = mid - 1
        # 没找到
        return False
```


