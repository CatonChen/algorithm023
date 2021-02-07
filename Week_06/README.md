# WEEK06学习笔记
### 基础知识
#### 动态规划
##### 基本思想
动态规划算法通常用于求解具有某种最优性质的问题。在这类问题中，可能会有许多可行解。每一个解都对应于一个值，我们希望找到具有最优值的解。动态规划算法与分治法类似，其基本思想也是将待求解问题分解成若干个子问题，先求解子问题，然后从这些子问题的解得到原问题的解。**与分治法不同的是，适合于用动态规划求解的问题，经分解得到子问题往往不是互相独立的。**若用分治法来解这类问题，则分解得到的子问题数目太多，有些子问题被重复计算了很多次。如果我们能够保存已解决的子问题的答案，而在需要时再找出已求得的答案，这样就可以避免大量的重复计算，节省时间。我们可以用一个表来记录所有已解的子问题的答案。不管该子问题以后是否被用到，只要它被计算过，就将其结果填入表中。这就是动态规划法的基本思路。具体的动态规划算法多种多样，但它们具有相同的填表格式。
##### 总结成以下几点
- 动态规划适用于具有重叠子问题和最优子结构性质的问题。
- 动态规划耗时比朴素解法少，如分治法，普通递归等。
- 动态规划思路，将大问题分解成不同部分的子问题，再根据子问题求得原问题解。
- 动态规划具有剪枝效果，因其对子问题仅求一次最优解；将子问题的解记忆化存储，遇到重复子问题时直接查表，降低时间复杂度。
##### 动态规划做题步骤
1. 明确dp[i]应该表示什么？（二维：dp[i][j]）
2. 根据dp[i]与dp[i-1]的关系得出状态转移方程
3. 确定初始条件，如dp[0]
##### leetcode动态规划代表题目（买卖股票系列）
- 121.买卖股票的最佳时机
- 122.买卖股票的最佳时机II
- 123.买卖股票的最佳时机III
- 188.买卖股票的最佳时机IV
- 309.最佳买卖股票时机含冷冻期
- 714.买卖股票的最佳时机含手续费

### 本周leetcode练习总结
#### 64.最小路径和
1. 动态规划
	1. 单元格(i,j)，只能从左边单元格(i-1,j)或上边单元格(i,j-1)走到，那么走到单元格(i,j)的最小路径和，等于左边单元格(i-1,j)或上边单元格(i,j-1)的较小者+当前单元格值gird[i][j]，得到动态规划方程：**dp[i][j]=min(dp[i][j-1],dp[i-1][j])+gird[i][j]**。具体分以下四种情况：
		1. 当i=j=0时，即起点，**dp[0][0]=gird[0][0]**
		2. 当i\<\>0,j\<\>0时，即不是上边界和左边界，**dp[i][j]=min(dp[i][j-1],dp[i-1][j])+gird[i][j]**
		3. 当i=0，j\<\>0时，即左边界，**dp[i][j]=dp[i][j-1]+grid[i][j]**
		4. 当i\<\>0，j=0时，即上边界，**dp[i][j]=dp[i-1][j]+grid[i][j]**
	2. 最终返回矩阵右下角，**dp[-1][-1]**
	3. 如果直接修改原矩阵单元格值，则不需要额外的矩阵空间
	4. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵
		2. 空间复杂度：O(1)，不需要额外空间
```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if i == j == 0:
                    grid[i][j] = grid[0][0]
                elif i == 0:
                    grid[i][j] = grid[i][j - 1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i - 1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i][j - 1], grid[i - 1][j]) + grid[i][j]
        return grid[-1][-1]
```

#### 122.买卖股票的最佳时机II
1. 贪心算法
	1. 假设第i天买和第i+1天卖，之间的利润差有持平、正数和负数三种情况。对于贪心算法，只累计正数的情况，最终得到的值即最大利润。
	2. 设**利润差tmp=price[i]-price[i-1]**，利润profit=0
		1. 当tmp\<=0时，跳过，即不买卖股票
		2. 当tmp\>0时，则将利润计入profit里，最终返回profit
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历整个数组
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            tmp = prices[i] - prices[i - 1]
            if tmp > 0:
                profit += tmp
        return profit
```

2. 动态规划
	1. 对于每天的交易结果，有两种可能的情况：截至目前为止，dp0当天持有股票时的最大利润和dp1当天不持有股票时的最大利润。那么得到状态转移方程：
		1. dp0=max(昨天不持有股票，昨天持有股票但今天卖出)
		2. dp1=max(昨天持有股票，昨天不持有股票但今天买入)
	2. 因每天最大利润只与前一天的最大利润有关，故状态可压缩，只通过dp0和dp1储存遍历的状态即可。
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历整个数组
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        dp0, dp1 = 0, float('-inf')
        for i in range(len(prices)):
            dp0 = max(dp0, dp1 + prices[i])
            # print(dp0,dp1)
            dp1 = max(dp1, dp0 - prices[i])
            # print(dp0,dp1)
        return dp0
```

#### 888.公平的糖果交换
1. 哈希表
	1. 计算两个数组差diff，将其中一个数组放入set集合中，遍历另一个数组，查找当前元素x-diff/2是否在set中存在。
	2. 复杂度分析
		1. 时间复杂度：O(n)，遍历一个数组
		2. 空间复杂度：O(n)，存储一个数组
```python
class Solution:
    def fairCandySwap(self, A: List[int], B: List[int]) -> List[int]:
        # 两个数组差
        diff = sum(A) - sum(B)
        set_B = set(B)
        for x in A:
            if x - diff // 2 in set_B:
                return [x, x - diff // 2]
```

#### 322.零钱兑换
1. 动态规划（自顶向下-递归）
	1. 利用**LRU\_cache**缓存已知结果
	2. 递归工作：
		1. 当金额\<0时，返回-1
		2. 当金额=0时，返回0
		3. 当金额\>0时，初始一个最少个数变量并初始化**mini=float(‘inf’)**
			1. 当剩余金额减去零钱面值（rem-coin）\>=0 且\<mini时，mini+1
			2. 否则即无法成立，返回-1
	3. 复杂度分析
		1. 时间复杂度：O(n)，n=金额\*硬币面值
		2. 空间复杂度：O(n)，lru缓存利用了额外空间
```python
import functools
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        @functools.lru_cache(amount)  # 缓存已知结果
        def dp(rem):
            if rem < 0:
                return -1
            elif rem == 0:
                return 0
            else:
                mini = float('inf')
                for coin in coins:
                    res = dp(rem - coin)
                    if res >= 0:
                        mini = min(mini, res + 1)  # 取较小者
            return mini if mini < float('inf') else -1

        return dp(amount)
```

2. 动态规划（自底向上-迭代）
	1. 当amount\<0时，返回-1
	2. 创建一个数组res，长度为**amount+1**
	3. 遍历数组res，遍历coins：
		1. 当i\>=coin时，使**res[i]=min(res[i],res[i-coin]+1)**
		2. 返回res[-1]
	4. 复杂度分析
		1. 时间复杂度：O(n)，res数组长度=金额大小，coins面值
		2. 空间复杂度：O(n)，res数组长度
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount < 0:
            return -1
        res = [0] + [amount + 1] * amount
        # print(res)
        for i in range(1, len(res)):
            for coin in coins:
                if i >= coin:
                    res[i] = min(res[i], res[i - coin] + 1)
        # print(res)
        return res[-1] if res[-1] != amount + 1 else -1
```

#### 127.单词接龙（本周复习）
1. 单向BFS
	1. 将wordList转换为**set**集合st
	2. 建立一个队列queue，**初始添加（beginWord,1）**
	3. 建立一个保存访问记录的**set**集合visited，初始添加beginWord
	4. 当queue不为空时，**取出首端元素赋值给cur和step**，cur表示字符串，step表示当前转换序列次数。
		1. 当cur==endWord，则返回step
		2. 当cur!=endWord，遍历cur每个字符，并用26个字母进行替换，生成tmp字符串。
		3. 当tmp字符串在st中存在时，则将tmp添加进queue且step+1，并在visited中记录tmp
	5. 当所有可能性都遍历过后，如果没有返回step，则返回0，表示不可能接龙成功。
	6. 复杂度分析
		1. 时间复杂度：O(m\*n)，m为单词长度，n为wordList个数
		2. 空间复杂度：O(n)，额外空间保存了记录个数
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        from collections import deque
        # 初始化
        st = set(wordList)

        if endWord not in st:
            return 0

        queue = deque()
        queue.append((beginWord, 1))
        visited = set(beginWord)

        while queue:
            cur, step = queue.popleft()
            if cur == endWord:
                return step
            for i in range(len(cur)):
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    tmp = cur[:i] + j + cur[i + 1:]
                    if tmp in st and tmp not in visited:
                        queue.append((tmp, step + 1))
                        visited.add(tmp)
        return 0
```

#### 213.打家劫舍II
1. 动态规划
	1. 环状排列的房屋，即首尾相接，那么只能在第一个或最后一个中偷窃其中一个。因此将环状排列约化为两个单排排列的问题：
		1. 偷第一个房子，不偷最后一个，为**nums[:-1]**，最大金额p1
		2. 偷最后一个房子，不偷第一个，为**nums[1:]**，最大金额p2
		3. 综合偷窃的最大金额，**Max(p1,p2)**
	2. 转移方程：
		1. 假设n间房子，前n间房子最高偷窃金额为**dp[n]**，前n-1间房子最高偷窃金额为**dp[n-1]**，此时再加一间房子，金额为**num**，则有：
			1. 不抢第n+1间房子，则**dp[n+1]=dp[n]**
			2. 抢n+1间房子，不抢第n间房子，则**dp[n+1]=dp[n-1]+num**
			3. 关于第n间房子是否被偷，存在两种可能：
				1. 假设第n间房子没有被偷，则dp[n]=dp[n-1]，那么**dp[n+1]=dp[n]+num=dp[n-1]+num**
				2. 假设第n间房子被偷，则**dp[n+1]=dp[n-1]+num**
		2. 最终方程为**dp[n+1]=max(dp[n],dp[n-1]+num)**
	3. 简化空间复杂度：因为dp[n]只与dp[n-1]和dp[n-2]有关系，所以通过变量cur和pre互相交替记录，即可把空间复杂度降为O(1)
	4. 复杂度分析
		1. 时间复杂度：O(n)，n为数组长度，需要遍历两次
		2. 空间复杂度：O(1)
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        # 递归
        def myRob(nums):
            cur = pre = 0
            for num in nums:
                cur, pre = max(pre + num, cur), cur
            return cur

        # 特例
        if len(nums) == 1:
            return nums[0]
        else:
            p1 = myRob(nums[:-1])
            p2 = myRob(nums[1:])
            return max(p1, p2)
```

#### 153.寻找旋转排序数组中的最小值（本周复习）
1. 二分查找
	1. 设置左边界left=0，右边界right=len(nums)-1，中位数mid=(left+right)//2
	2. 当left\<right时，有以下几种情况：
		1. 当nums[0]\<nums[mid]，则表示目标在nums[0]～nums[mid]之间，移动right=mid-1，
		2. 当nums[0]\>nums[mid]，则表示目标在nums[mid]～nums[right]之间，移动left=mid+1
		3. 当nums[mid]\>nums[mid+1]成立，则nums[mid+1]为最小元素
		4. 当nums[mid-1]\>nums[mid]成立，则nums[mid]为最小元素
	3. 复杂度分析：
		1. 时间复杂度：O(logn)
		2. 空间复杂度：O(1)
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:  # 特例1
            return nums[0]
        left, right = 0, len(nums) - 1
        if nums[right] > nums[left]:  # 特例2，是个生序数组
            return nums[0]
        # 二分查找
        while left < right:
            mid = (left + right) // 2
            # print (left,right,mid)
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            if nums[mid - 1] > nums[mid]:
                return nums[mid]
            if nums[left] < nums[mid]:
                left = mid + 1
            else:
                right = mid - 1
```

#### 198.打家劫舍
1. 动态规划
	1. 转移方程：
		1. 假设n间房子，前n间房子最高偷窃金额为**dp[n]**，前n-1间房子最高偷窃金额为**dp[n-1]**，此时再加一间房子，金额为**num**，则有：
			1. 不抢第n+1间房子，则**dp[n+1]=dp[n]**
			2. 抢n+1间房子，不抢第n间房子，则**dp[n+1]=dp[n-1]+num**
			3. 关于第n间房子是否被偷，存在两种可能：
				1. 假设第n间房子没有被偷，则dp[n]=dp[n-1]，那么**dp[n+1]=dp[n]+num=dp[n-1]+num**
				2. 假设第n间房子被偷，则**dp[n+1]=dp[n-1]+num**
		2. 最终方程为**dp[n+1]=max(dp[n],dp[n-1]+num)**
	2. 简化空间复杂度：因为dp[n]只与dp[n-1]和dp[n-2]有关系，所以通过变量cur和pre互相交替记录，即可把空间复杂度降为O(1)
	3. 复杂度分析
		1. 时间复杂度：O(n)，n为数组长度，需要遍历两次
		2. 空间复杂度：O(1)
```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        cur = pre = 0
        for num in nums:
            cur, pre = max(pre + num, cur), cur
        return cur
```

#### 589.N叉树的前序遍历（本周复习）
1. 递归法
	1. 终止条件：当没有节点时，返回终止
	2. 遍历节点的孩子节点，将孩子节点依次调用自身函数
	3. 返回最终结果
	4. 复杂度分析
		1. 时间复杂度：O(n)，遍历树的所有节点
		2. 空间复杂度：O(n)，递归工作时调用的系统栈
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""

class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res=[]
        #终止条件
        if root is None:
            return
        res.append(root.val)
        for n in root.children:
            res.extend(self.preorder(n))
        return res
```

2. 迭代法
	1. 定义一个队列**queue**，初始时将root压入队列
	2. 当队列不为空时，弹出首端元素加入结果，同时将孩子节点**逆序**逐个压入队列，不断迭代至队列为空为止。
	3. 返回最终结果
	4. 复杂度分析
		1. 时间复杂度：O(n)，遍历树的所有节点
		2. 空间复杂度：O(n)，维护了一个队列，保存了所有节点
```python
"""
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children
"""
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        if not root:
            return res
        queue = []
        queue.append(root)
        while queue:
            node = queue.pop()
            res.append(node.val)
            for n in node.children[::-1]:
                queue.append(n)
        return res
```

#### 169.多数元素（本周复习）
1. pythonic
	1. 根据题解多数元素指出现次数大于**[n/2]**的元素
	2. 将数组排序后，出现次数大于[n/2]的元素必然出现在[n/2]的位置上
	3. 复杂度分析
		1. 时间复杂度：O(logn)，数组排序时间
		2. 空间复杂度：O(1)
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        return sorted(nums)[len(nums)//2]
```

1. 字典（哈希表）
	1. 建立一个字典**dict**
	2. 当dict不存在该元素的时候，将元素加入字典作为key，同时count=1作为val；否则将已有的元素val+1
	3. 当数组遍历完毕后，找出字典中**val\>n/2**的元素返回
	4. 复杂度分析
		1. 时间复杂度：O(n)，遍历数组，遍历字典
		2. 空间复杂度：O(n)，字典利用了额外空间
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dict = {}
        for num in nums:
            if dict.get(num) is None:
                dict[num] = 1
            else:
                dict[num] += 1
        tmp = len(nums) // 2
        for k, v in dict.items():
            if v > tmp:
                return k
```

#### 121.买卖股票的最佳时机
1. 动态规划
	1. 维护一个最小价格minprice，初始minprice=price[0]
	2. 维护一个长度为n的数组dp，n为price的长度
	3. 因为dp[i]的最大利润，为前一天的最大利润dp[i-1]，与当天价格price减去minprice记录的最小价格之间的利润，之间的较大者，故得到状态转移方程：dp[i]=max(dp[i-1],price[i]-minprice)
	4. 最终返回dp[i]
	5. 复杂度分析
		1. 时间复杂度：O(n)，遍历整个数组
		2. 空间复杂度：O(n)，dp数组大小为数组price的大小
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        if n == 0:
            return 0
        dp = [0] * n
        minprice = prices[0]

        for i in range(1, n):
            minprice = min(minprice, prices[i])
            dp[i] = max(dp[i - 1], prices[i] - minprice)

        return dp[-1]
```

2. 一次遍历
	1. 在上述动态规划的方法上进行改进，通过minprice和maxprofit记录到当天为止时的最小价格和最大利润
	2. 初始minprice=float(‘inf’)，maxprofit=0
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历一次数组
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        # 初始化最小价格和最大利润
        minprice = float('inf')
        maxprofit = 0
        for price in prices:
            minprice = min(price, minprice)
            maxprofit = max(maxprofit, price - minprice)
        return maxprofit
```

#### 363.矩形区域不超过k的最大数值和
1. 前缀和+二分查找
	1. 算法思路：
		1. 固定左右边界
		2. 求得每行的总和，通过前缀和找与k的最大差值来确定最大矩阵
	2. 算法实现
		1. 通过左右边界，求得每行总和
		2. 通过二分查找（bisect），找出不超过k的矩阵
	3. 复杂度分析
		1. 时间复杂度：O(m\*n\*k)
		2. 空间复杂度：O(m\*n)
```python
class Solution:
    def maxSumSubmatrix(self, matrix: List[List[int]], k: int) -> int:
        import bisect  # 二分查找模块
        row = len(matrix)  # 行
        col = len(matrix[0])  # 列
        res = float('-inf')
        for left in range(col):
            # left边界下的每行总和
            _sum = [0] * row
            # print(_sum)
            for right in range(left, col):
                # print(right)
                for j in range(row):
                    _sum[j] += matrix[j][right]
                    # print(_sum)
                arr = [0]  # 中间数组
                cur = 0
                for tmp in _sum:
                    cur += tmp  # 前缀和
                    # 二分查找，找出最接近cur-k的下标
                    loc = bisect.bisect_left(arr, cur - k)
                    if loc < len(arr):
                        res = max(res, cur - arr[loc])  # 将最接近k的前缀和赋予res
                    bisect.insort(arr, cur)  # 将前缀和放入res并排序
                    # print(arr)
        return res
```

#### 33.搜索旋转排序数组（本周复习）
1. 暴力法
	1. 遍历数组，找到匹配目标的元素时，返回下标，否则返回-1.
	2. 复杂度分析
		1. 时间复杂度：O(n)，n为数组长度
		2. 空间复杂度：O(1)
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        for i in range(len(nums)):
            if nums[i] == target:
                return i
        return -1
```

2. 二分查找
	1. 套用二分查找模板，**left=0,right=len(nums)-1,mid=(left+right)/2**
	2. 通过mid将数组一分为二，每次只查找一半数组的长度，通过left,right和mid的值比较来判断丢弃哪一半。
	3. 当left\<right时，分三种情况：
		1. **当nums[mid]==target，则返回mid**
		2. **当nums[mid]\>nums[left]，表示nums[left]\~nums[mid]是有序数组**，这时考虑：
			1. 当nums[left]\<=target\<nums[mid]时，移动right=mid-1，寻找目标；否则移动left=mid+1
		3. **当nums[mid]\<nums[left]，表示nums[mid]\~nums[right]是有序数组**，这时考虑：
			1. 当nums[mid]\<target\<=nums[right]时，移动left=mid+1，寻找目标；否则移动right=mid-1
	4. 复杂度分析
		1. 时间复杂度：O(logn)，二分查找的时间复杂度
		2. 空间复杂度：O(1)
	5. 题解一
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[left] < nums[mid]:  # 左边有序
                if nums[mid] < target:  # target在mid右边
                    left = mid + 1
                elif nums[mid] > target:  # target在mid左边
                    if target >= nums[left]:  # target大于等于左边界
                        right = mid - 1
                    else:
                        left = mid + 1
            elif nums[left] > nums[mid]:  # 右边有序
                if nums[mid] > target:  # target在mid左边
                    right = mid - 1
                else:  # target在mid右边
                    if target < nums[left]:
                        left = mid + 1
                    else:
                        right = mid - 1
            else:
                left += 1
        # 没有目标
        return -1
```

6. 题解二
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < nums[right]:  # mid~right是升序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            else:  # left~mid是升序
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
        if nums[left] == target:	#当left=target
            return left
        else:
            return -1
```

#### 200.岛屿数量（本周复习）
1. 深度优先搜索（DFS）
	1. 思路：**当遇到1的时候，岛屿数量+1，然后将这个1周围的所有1（含间接的1）都置为0，之后继续搜索下一个1，直到遍历完成**。
	2. 算法过程
		1. 特例：当gird为空时，返回0
		2. 双指针循环遍历gird，i遍历行，j遍历列，当gird[i][j]==1时，岛屿数量+1，同时进入递归工作，将这个1其周围的1（含间接连续的1）都置为0
		3. 具体递归工作实现：
			1. 终止条件：
				1. **当i\<0 or j\<0 or i \>=len(gird) or j\>=len(grid[0])时终止，表示i,j坐标已超出grid范围。**
				2. **当gird[i][j]!=1时，表示周边没有需要置为0的1了**
			2. 持续递归工作，直到满足递归终止条件。
	3. 复杂度分析
		1. 时间复杂度：O(m\*n)，gird的行列乘积
		2. 空间复杂度：O(m\*n)，递归需要遍历所有整个矩阵
```python
class Solution:
    def dfs(self, grid, i, j):
        # 超出grid边界，或当前坐标不为1时，终止
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return
        grid[i][j] = '0'  # 将当前坐标置为0，且将其周围坐标也置为0
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i + 1, j)

    def numIslands(self, grid: List[List[str]]) -> int:
        if grid is None:
            return 0

        m = len(grid)
        n = len(grid[0])
        count = 0
        # 双指针遍历grid
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)  # 递归工作
                    count += 1

        return count
```

#### 367.有效的完全平方数（本周复习）
1. 二分查找
	1. n\<2时，返回true
	2. 设**left=2, right=num/2，当left\<=right时，x=(left+right)/2**，判断x\*x==num是否成立。
		1. 当x\*x==num时，返回true
		2. 当x\*x\>num时，right=x-1
		3. 当x\*x\<num时，left=x+1
		4. 如果全部不符合，返回false
	3. 复杂度分析
		1. 时间复杂度：O(logn)，二分查找时间复杂度
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True

        left = 2
        right = num // 2
        while left <= right:
            x = (left + right) // 2
            if x * x == num:
                return True
            elif x * x > num:
                right = x - 1
            else:
                left = x + 1
        return False
```

2. 牛顿迭代法
	1. 公式：**x=1/2\*(x+num/x)**
	2. 初始，x=num/2。
	3. 当x\*x\>num时，不停循环迭代，根据公式调整x的值，直到循环终止。
	4. 返回 x\*x==num的布尔结果
	5. 复杂度分析
		1. 时间复杂度：O(logn)
		2. 空间复杂度：O(1)
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        if num < 2:
            return True

        # 牛顿迭代法
        x = num // 2
        while x * x > num:
            x = (x + num // x) // 2
        return x * x == num
```

#### 212.单词搜索II
1. Trie前缀树+回溯
	1. 创建一个**Trie字典树**，将words里的单词构建成一个Trie，后序用于匹配。
	2. 遍历board每个单元格，如果字典中存在以单元格字母为开头的单词，则开始进行回溯探索**backtracking(cell)**。递归工作步骤如下：
		1. 将Trie里所有以单元格的字母开头的单词返回出来currnode，进行匹配，将匹配到的单词加入结果中。
		2. 将当前单元格字母置为已使用，同时对该单元格的上下左右相邻单元格进行递归工作。递归**终止条件**有两种可能：
			1. 当单元格坐标超出board的范围，则返回
			2. 当单元格字母不在currnode中，则返回
		3. 递归后进行回溯，将当前单元格的字母恢复。
		4. 对在currnode中遍历过的字母进行剪枝，减少重复计算。
	3. 在递归函数backtracking(cell)中，同时探索当前单元格的相邻单元格**backtracking(neighborcell)**。在每次递归工作时，都会检查到目前为止的字母序列是否与字典中的任何单词匹配。
	4. 复杂度分析
		1. 时间复杂度：O(m\*n)，m是board中所有单元格个数，n为单词长度
		2. 空间复杂度：O(n)，字典树Trie的总数
```python
class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        # 创建字典树
        WORD_KEY = '$'
        trie = {}
        for word in words:  # 单词
            node = trie
            for letter in word:  # 字母
                node = node.setdefault(letter, {})
            node[WORD_KEY] = word  # 单词用$标记出来
            # print(node)
        # print(trie)

        rownum = len(board)
        colnum = len(board[0])
        matchedWords = []

        # 递归回溯
        def backtracking(row, col, parent):
            letter = board[row][col]
            # print(parent)
            # print(letter)
            currnode = parent[letter]
            # print(currnode)
            word_match = currnode.pop(WORD_KEY, False)
            # print(word_match)
            if word_match:
                matchedWords.append(word_match)

            # 将字母标记为#，即已使用
            board[row][col] = '#'
            # 探索当前单元格四周单元格 上右下左
            for (x, y) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                new_x, new_y = row + x, col + y
                if new_x < 0 or new_y < 0 or new_x >= rownum or new_y >= colnum:
                    continue
                if not board[new_x][new_y] in currnode:
                    continue
                backtracking(new_x, new_y, currnode)
            # 回溯，恢复状态
            board[row][col] = letter
            # 剪枝：当前currnode已经遍历过了，从父节点中删除
            if not currnode:
                parent.pop(letter)

        # 遍历board的所有单元格
        for row in range(rownum):
            for col in range(colnum):
                if board[row][col] in trie:
                    # print(board[row][col])
                    backtracking(row, col, trie)

        return matchedWords
```

#### 455.分发饼干（本周复习）
1. 贪心算法
	1. 根据条件s[i]\>=g[j]，才能达成饼干i满足孩子胃口j，故先对数组s,g排序。
	2. 排序后，当s[i]\>=g[j]，则结果+1，同时i,j各加一；否则只移动i，寻找下一块饼干
	3. 复杂度分析
		1. 时间复杂度：O(logn)，两次排序，2\*logn的时间
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # 排序
        g.sort()
        s.sort()
        i = j = res = 0

        while i < len(s) and j < len(g):
            if s[i] >= g[j]:
                res += 1
                i += 1
                j += 1
            else:
                i += 1

        return res
```

#### 665.非递减数列
1. 迭代
	1. 遍历一次数组，当出现nums[i]\<nums[i-1]时，即破坏了数组的单调递增性，为了保持数组的有序性，考虑以下三种可能：
		1. 当i=1时，将nums[i-1]修改为nums[i]
		2. 当i\>1时，且nums[i]\>=nums[i-2]，将nums[i-1]修改为nums[i]
		3. 当i\>1时，且nums[i]\<nums[i-2]，将nums[i]修改为nums[i-1]
	2. 用一个计数器count记录修改的次数，最后返回count\<=1的布尔值
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历一次数组
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                if i == 1:
                    nums[i - 1] = nums[i]
                elif i > 1:
                    if nums[i] >= nums[i - 2]:
                        nums[i - 1] = nums[i]
                    elif nums[i] < nums[i - 2]:
                        nums[i] = nums[i - 1]
                count += 1
            # print(count)
        return True if count < 2 else False
```
4. 代码优化：
```python
class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        count = 0
        for i in range(1, len(nums)):
            if nums[i] < nums[i - 1]:
                count += 1
                if i == 1 or nums[i] >= nums[i - 2]:
                    nums[i - 1] = nums[i]
                else:
                    nums[i] = nums[i - 1]
        return count < 2
```


