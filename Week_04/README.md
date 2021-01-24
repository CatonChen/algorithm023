# WEEK04学习笔记
### 基础知识
#### 深度优先搜索（DFS）和广度优先搜索（BSF）
搜素即遍历，就是对树（图）的所有节点遍历一次，且每个节点仅访问一次，最终找到目标结果。根据对节点的访问顺序，分为深度优先和广度优先。
##### 深度优先搜索
DFS主要通过递归方法（或手动维护栈）实现。
- 如果是二叉树，那么从根节点开始，依次递归访问每个节点的左节点和右节点，直到遍历完所有节点。
- 如果是多叉树，同样从根节点开始，依次递归访问每个节点的孩子节点，直到遍历完所有节点。
###### DFS代码模板
1. 递归代码模板
```python
#Python
visited = set() 
def dfs(node, visited):    
	if node in visited: # terminator    	
	# already visited     	
	return 	
	visited.add(node) 	
	# process current node here. 	
	...	
	for next_node in node.children(): 		
		if next_node not in visited: 			
			dfs(next_node, visited)
```

2. 非递归代码模板（栈）
```python
#Python
def DFS(self, tree): 	
	if tree.root is None: 		
		return [] 	
	visited, stack = [], [tree.root]	
	while stack: 		
		node = stack.pop() 		
		visited.add(node)		
		process (node) 		
		nodes = generate_related_nodes(node) 
		stack.push(nodes) 	
	# other processing work 	
	...
```

##### 广度优先搜索
BFS通常是通过维护队列来完成对所有节点的搜索，其利用的是队列先进先出的特性来处理当前层的节点。类似于水波纹逐层从左往右访问各个节点。
以二叉树说明：
- 一开始将根节点root加入队列queue，取出它，对其进行处理，同时将根节点的左右子节点压入队列（等同于将下一层所有节点压入队列）。
- 类似于循环取出当前层的节点进行处理，并同时将下一层的节点按从左往右的顺序压入队列，反复迭代至所有节点都处理完毕。
	![]()
###### BFS代码模板
```python
# Python
def BFS(graph, start, end):    
	visited = set()	
	queue = [] 	
	queue.append([start]) 	
	while queue: 		
		node = queue.pop() 		
		visited.add(node)		
		process(node) 		
		nodes = generate_related_nodes(node) 
		queue.push(nodes)	
	# other processing work 	
	...
```

#### 贪心算法
- 贪心算法是一种在每一步选择中都采取在当前状态下最好或最优(即最有 利)的选择，从而希望导致结果是全局最好或最优的算法。
- 贪心算法与动态规划的不同在于它对每个子问题的解决方案都做出选择，不 能回退。动态规划则会保存以前的运算结果，并根据以前的结果对当前进行 选择，有回退功能。
	> 贪心：当下做局部最优
	> 回溯：回退上一步
	> 动态规划：最优子问题+回退
适用贪心算法的场景：简单地说，问题能够分解成子问题来解决，子问题的最优解能递推到最终 问题的最优解。这种子问题最优解称为最优子结构。

#### 二分查找
##### 二分查找的前提
1. 单调性（递增或递减）
2. 存在上下界（或左右界）
3. 能够通过索引查询
##### 二分查找代码模板
```python
left, right = 0, len(array) - 1 
while left <= right:
   mid = (left + right) / 2
   if array[mid] == target:
      	# find the target!!
      	break or return result
   elif array[mid] < target:
     	left = mid + 1
   else:
		right = mid - 1
```

### 本周leetcode练习总结
#### 46.全排列（本周复习）
给定一个没有**重复数字**的序列，返回其所有可能的全排列
1. 回溯法
	1. 主要以DFS（深度优先搜索）的递归方法为主干；对遍历过的节点增加状态标识，向下选择的时候打上“使用”标识，向上回溯的时候打上“撤销”标识，以便元素可以再次被使用。
	2. 如下图所示，按DFS搜索顺序，依次选择数字，当前已选择的数字不能在要选择的数字中出现，所以状态标识used打true，后续为生成其他组合，所以在回溯时used打false，以便在其他组合中可以继续使用。
		![]()
	3. 递归说明
		1. 定义变量
			1. nums，传入的数字序列
			2. size，nums的长度
			3. depth，递归深度，表示递归到第几层，初始为0
			4. path，保存一个排列组合的中间结果的变量
			5. used，数字状态的标识，true或false
			6. res，最终结果
		2. 递归终止条件：一个排列的数字够了就可以把结果保存进res；当depth的深度等于nums的长度时，就表示排列的数组的长度够了，因为输入序列nums为[1,2,3]，递归到第4层时，depth=3。
		3. 递归工作
			1. 非根节点或叶子节点，其余节点都要做递归工作
			2. 当当前数字节点未被使用时，把used状态标为true，并将数字添加进path中保存，然后调用递归函数进行递归工作（这里depth+1）。
			3. 当递归函数调用后，要重置该数字的状态（used状态标为false），同时撤销该数字元素（path.pop）
	4. 复杂度分析
		1. 时间（空间）复杂度：O(n\*n!)，回溯通常为指数级复杂度。
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums,size,depth,path,used,res):
            #terminator
            if depth==size:
                res.append(path.copy())
                return
            #current logic
            for i in range(size):
                if not used[i]:
                    used[i]=True
                    path.append(nums[i])
                    #recursion    
                    dfs(nums,size,depth+1,path,used,res)
                    #reserve statu
                    used[i]=False
                    path.pop()

        res=[]
        size = len(nums)
        used = [False for _ in range(size)]
        dfs(nums,size,0,[],used,res)
        return res
```

#### 1.两数之和（本周复习）
1. 字典（hash）
	1. 两个数a和b，及目标值target，利用公式b=target-a来检测字典中是否有相应的b，如果b不存在于字典中，就添加进字典，继续遍历下一个元素，如果找到就返回达成条件的a、b下标
	2. 复杂度分析
		1. 时间复杂度：O(n)，n为数组长度
		2. 空间复杂度：O(n)，取决于保存了多少元素，最差就是保存数组的所有元素
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap=dict()
        res = []
        #枚举
        for i , v in enumerate(nums):
            if hashmap.get((target-v)) is not None:
                    res.append(i)
                    res.append(hashmap.get(target - v))
            hashmap[v]=i
        return res
```

#### 860.柠檬水找零
1. 贪心算法
	1. 记录5和10两个数，five和ten两个变量
	2. 当bill是5时，five+1；当bill是10时，如果five\>0，则five-1，ten+1，如果不满足，返回false
	3. 当bill是20时，两种情况
		1. 当ten\>0 且five\>0， 则ten-1 ， five-1
		2. 当five\>2时，则five-3
		3. 如果上面两种都不满足，则返回false
	4. 如果遍历完数组，就返回true
	5. 复杂度分析
		1. 时间复杂度：O(n)，需要遍历完数组
		2. 空间复杂度：O(1)，保存常
```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five , ten = 0 , 0
        for bill in bills:
            if bill == 5:
                five+=1
            elif bill == 10:
                if five>0:
                    five-=1
                    ten+=1
                else:
                    return False
            elif bill == 20:
                if five>0 and ten>0:
                    five-=1
                    ten-=1
                elif five>2:
                    five-=3
                else:
                    return False
        return True
```

#### 70.爬楼梯（本周复习）
1. 递归
	1. 根据一次只能一级台阶或二级台阶，推导可知f(n)=f(n-1)+f(n-2)
	2. 注意两点：
		1. n=1或n=2时，直接返回
		2. 当n很大的时候，递归计算十分耗时，所以需要lru\_cache缓存装饰器来保存已经计算过的值，减少重复计算的过程耗时。
	3. 复杂度分析
		1. 时间复杂度：O(2^n)，添加lru缓存会减少时间复杂度到O(n)
		2. 空间复杂度：O(n)，取决于系统调用的递归栈
```python
class Solution:
    @functools.lru_cache(100)
    def climbStairs(self, n: int) -> int:
        #递归
        if n==1 or n==2:
            return n
        return self.climbStairs(n-1)+self.climbStairs(n-2)
```

2. 迭代
	1. 利用两个指针a,b表示n-1,n-2，利用变量tmp保存结果。
	2. 循环n次，每次循环让tmp=a+b,a=b,b=tmp，达到滚动数组的效果，最终返回tmp。
	3. 复杂度分析
		1. 时间复杂度：O(n)，取决于n的大小
		2. 空间复杂度：O(1)，常量变量
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        #  当n==1或2时，返回n
        if n == 1 or n == 2:
            return n
        # 初始化变量
        a, b, tmp = 1, 2, 0
        for i in range(3, n + 1):
            # print(i)
            tmp = a + b
            a = b
            b = tmp
        # 返回结果
        return tmp
```

#### 15.三数之和（本周复习）
1. 双指针+排序
	1. 先将数组nums排序，以避免重复解。
	2. 定义k,i,j三个变量表示数组元素下标，且有k\<i\<j，初始k=0, i=k+1, j = len(nums)-1，这样i,j为左界和右界。
	3. 外循环k in range(len(nums)-2) ， 当i\<j时内循环，寻求nums[k]+nums[i]+nums[j]=0的三元组s。其中：
		1. 当s\<0时，即i太小，需移动左界i+=1,并判断是否nums[i]==nums[i-1]，如相等继续移动左界i。
		2. 当s\>0时，则j太大，需移动右界j-=1,并判断是否nums[j]=nums[j+1]，如相等继续移动右界j。
		3. 当s=0时，找到合适的三元组，保存至结果数组res中，并且左右界各自移动一步，且还需判断上述1和2点。
		4. 考虑一个特殊点：如果nums[k]\>0，且数组已排序和因为k\<i\<j，所以nums[i]和nums[j]都\>0，故不存在nums[k]+nums[i]+nums[j]=0的可能，所以直接返回空解。
	4. 复杂度分析
		1. 时间复杂度：O(n^2)，内外循环均需要遍历数组n
		2. 空间复杂度：O(1)，指针变量为常数变量
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        res = []
        # order array
        nums.sort()
        # outer loop
        for k in range(n - 2):
            if nums[k] > 0:  # nums[k]>0, 意味没有解
                break
            # 如果k和k-1的元素相等，继续移动k
            if k > 0 and nums[k] == nums[k - 1]:
                k += 1
                continue
            i, j = k + 1, n - 1
            # inner loop
            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s < 0:  # i太小，移动i
                    i += 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                elif s > 0:  # j太大，移动j
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                else:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
        
        return res
```

#### 122.买卖股票的最佳时机II
1. 贪心算法
	1. 股票买卖策略
		1. 单个交易日：假设今天价格p1，明天价格p2，那么今天买，明天卖就有p2-p1的利润（负值即亏损）。
		2. 价格连续上涨p1,p2,p3…pn，pn-p1的利润最大，等同于每天买卖profit=(P2-P1)+(P3-P2)+…(Pn-Pn-1)=Pn-P1
		3. 价格连续下降，则不交易，避免亏损
	2. 算法推导
		1. 遍历整个股票价格列表price，策略是上涨的交易日都买卖（赚取利润），下跌的交易日都不买卖（避免亏损），返回最后的利润之和。
		2. 设tmp=price[i]-price[i-1]，即i日卖出和i日买入的价差利润，当tmp\>0是计入profit，tmp\<=0则跳过，最终返回profit。
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历一次数组price
		2. 空间复杂度：O(1)，常量变量
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        for i in range(1, len(prices)):
            # i日卖出与i-1日买入的利润
            tmp = prices[i] - prices[i - 1]
            if tmp > 0:  # 利润>0，则计入总利润
                profit += tmp
        return profit
```

#### 22.括号生成（本周复习）
1. 回溯（实质是种递归）
	目标是生成n对的括号，即（）算一对括号，所以要达成以下递归工作条件：
	1. 终止条件：当左右括号的个数=n时，就返回结果
	2. 要先有左括号，后有右括号，才能得到有效的括号对，所以：
		1. 左括号随时可以添加，但要保证left个数\<n成立。
		2. 右括号要在有左括号之后才可以添加，等同于当前左括号使用个数left\>当前右括号使用个数right。
2. 复杂度分析
	1. 时间复杂度：O(n)，因为左右括号都要递归n次，所以为O(2n)=O(n)
	2. 空间复杂度：O(n)，取决于递归时系统栈
```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # dfs 回溯
        def dfs(cur_str, left, right):
            # 递归终止
            if left == n and right == n:
                return res.append(cur_str)
            # 处理逻辑与下钻
            if left < n:
                dfs(cur_str + '(', left + 1, right)
            if left > right:
                dfs(cur_str + ')', left, right + 1)

        res = []
        dfs('', 0, 0)
        return res
```

#### 422.分发饼干
1. 贪心算法
	1. 因为有s[j]\>=g[i]即满足一个孩子，所以先对s、g排序。
	2. 同时遍历已排序的s、g，当s[j]\>=g[i]时，满足+1，否则移动j在s中寻找下一块符合条件的饼干。
	3. 复杂度分析
		1. 时间复杂度：O(n)，n为数组s+g
		2. 空间复杂度：O(1)，常量指针
```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # 排序 g,s
        g.sort()
        s.sort()
        res = 0
        i, j = 0, 0
        while i < len(g) and j < len(s):
            if s[j] >= g[i]:  # 满足条件 res+1，移动s,g的下标
                res += 1
                i += 1
                j += 1
            else:
                j += 1  # 否则移动s下标
        return res
```

#### 860.柠檬水找零（本周复习）
1. 贪心算法
	1. 定义five和ten两个变量记录零钱数量
	2. 推导是否可以找零的过程：
		1. 如果是5元，则five+1
		2. 如果是10元：a.如果five\>0，则five-1, ten+1；不满足返回false 
		3. 如果是20元：a.如果ten\>0 且 five\>0 ，则ten-1 , five-1；b.如果five\>2，则five-3 ；不满足返回false
		4. 如果上述1和2直到序列遍历结束都成立，则返回true
	3. 复杂度分析
		1. 时间复杂度：O(n)，遍历整个数组
		2. 空间复杂度：O(1)，常量变量
```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        five, ten = 0, 0
        for bill in bills:
            if bill == 5:
                five += 1
            elif bill == 10:
                if five > 0:
                    five -= 1
                    ten += 1
                else:
                    return False
            else:
                if ten > 0 and five > 0:
                    ten -= 1
                    five -= 1
                elif five > 2:
                    five -= 3
                else:
                    return False
        return True
```

#### 42.接雨水（本周复习）
1. 动态规划
	1. 通过maxleft和maxright两个数组来保存下标i的左边界最大高度和右边界最大高度
	2. 一遍遍历数组，利用下标i的maxleft和maxright中的较小者和hight[i]的高度差来计算可以接到的雨水量。
%% 测试用例：   [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
%% 左边界数组： [0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]
%% 右边界数组： [3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 1]
%%    较小者： [0, 1, 1, 2, 2, 2, 2, 3, 2, 2, 2, 1]
3. 复杂度分析
	1. 时间复杂度：O(n)，只需要遍历三次数组，O(3\*n)=O(n)
		2. 空间复杂度：O(n)，需要两个数组保存maxleft和maxright
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 没有高度，返回0
        if not height:
            return 0
        n = len(height)
        res = 0
        # 左右边界
        maxleft = [0] * n  # 左边界数组
        maxright = [0] * n  # 右边界数组
        maxleft[0] = height[0]
        maxright[n - 1] = height[n - 1]
        # print(height)
        # tmp = []
        # 左边界数组最大值
        for i in range(1, n):
            maxleft[i] = max(height[i], maxleft[i - 1])
        # print(maxleft)
        # 右边界数组最大值
        for i in range(n - 2, -1, -1):
            maxright[i] = max(height[i], maxright[i + 1])
        # print(maxright)
        # 一次遍历数组，通过高度差得知雨水量
        for i in range(n):
            # tmp.append(min(maxleft[i], maxright[i]))
            if min(maxleft[i], maxright[i]) > height[i]:
                res += min(maxleft[i], maxright[i]) - height[i]
        # print(tmp)
        return res
```

#### 874.模拟行走机器人
1. 贪心算法
	1. 建立上下左右的字典
	2. 将障碍数组obstacles用set处理（很重要！）
	3. 进行指令模拟：
		1. 当command=-2时左转
		2. 当command=-1时右转
		3. 当command\>0时根据command代表的长度行走，碰到障碍坐标时中断
	4. 复杂度分析
		1. 时间复杂度：O(n)，遍历指令数组的长度
		2. 空间复杂度：O(n)，利用字典保存指令
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 定义方向字典
        direction = {
            'up': [0, 1, 'left', 'right'],
            'down': [0, -1, 'right', 'left'],
            'left': [-1, 0, 'down', 'up'],
            'right': [1, 0, 'up', 'down']
        }

        # 初始化
        x, y = 0, 0
        res = 0
        dir = 'up'

        # 处理障碍物
        obstacles = set(map(tuple, obstacles))

        # 指令模拟
        for cmd in commands:
            if cmd == -1:  # 右转
                dir = direction[dir][3]
            elif cmd == -2:  # 左转
                dir = direction[dir][2]
            else:  # 正常指令
                for i in range(cmd):
                    if (x + direction[dir][0], y + direction[dir][1]) in obstacles:
                        break
                    else:
                        x += direction[dir][0]
                        y += direction[dir][1]
                        res = max(res, x ** 2 + y ** 2)

        return res
```

2. 贪心算法
	1. 与上述不同的在于利用元组表示上下左右的坐标，上下左右依次为[(0,1),(0,-1),(-1,0),(1,0)]()，形成一种顺时针方向闭环。
		 ![]()
	2. 假设di=0，为初始方向，则左转：（di+3）%4 ， 右转（di+1）%4
	3. 其他指令模拟类似上一个题解思路
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 定义方向
        x_y = ((0, 1), (1, 0), (0, -1), (-1, 0))

        # 初始化
        x, y = 0, 0
        res = 0
        dir = 0

        # 处理障碍物
        obstacles = set(map(tuple, obstacles))
        # print(obstacles)
        # 指令模拟
        for cmd in commands:
            if cmd == -1:  # 右转
                dir = (dir + 1) % 4
            elif cmd == -2:  # 左转
                dir = (dir + 3) % 4
            else:  # 正常指令
                for i in range(cmd):
                    dx, dy = x_y[dir]
                    next_x = x + dx
                    next_y = y + dy
                    # print(next_x, next_y)
                    if (next_x, next_y) in obstacles:
                        break
                    x, y = next_x, next_y
                    res = max(res, x ** 2 + y ** 2)

        return res
```

#### 200.岛屿数量
1. 深度优先搜索（DFS）
	目的：求解标记为1的连续区域的个数。
	1. 当碰到1的时候，岛屿数量+1，然后将这个1其周围的1都置为0（包含周围1的周围的1），直到没有1为止，然后寻找下一个1。
	2. 主要算法过程
		1. 当grid为空时，返回0
		2. 双指针循环，一个指针i按gird的横轴遍历，一个指针j按gird的纵轴遍历，当gird[i][j]==‘1’时，岛屿数量+1，同时进入dfs递归工作，将这个1及其周围的1都标记为0。
		3. 递归工作
			1. 终止条件有两：
				1. 当i\<0 或j\<0 或 i\>=gird横轴长度 or j\>=gird纵轴长度 ， 则表示i,j坐标超出了gird范围
				2. 当gird[i][j]不等于1时，表示附近没有需要置为0的1了
			2. 继续通过递归工作，将这个1附近的1置为0，直到满足递归终止条件。
	3. 复杂度分析
		1. 时间复杂度：O(m\*n)，m是gird的横轴长度，n是gird的纵轴长度
		2. 空间复杂度：O(m\*n)，需要递归m\*n次。
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # 处理gird为空时的特例
        if not grid:
            return 0

        # grid的x,y轴
        m = len(grid)
        n = len(grid[0])
        count = 0
        # 遍历grid
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.dfs(grid, i, j)
                    count += 1

        return count

    def dfs(self, grid, i, j):
        # 终止条件
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != '1':
            return
        # 递归工作
        grid[i][j] = '0'
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
```

#### 989.数组形式的整数加法
1. 数组变换
	根据a=[1,2,0,0] 和 k=34，则有 1200+34=1234，期望输出[1,2,3,4]
	1. 利用数据类型转换，将数组a转为int类型，+k后，再加int类型转为string类型，最后将其数组化返回即可
	2. 本题和**66.加一**解法很相似
	3. 复杂度分析
		1. 时间复杂度：O(n)
		2. 空间复杂度：O(1)
```python
class Solution:
    def addToArrayForm(self, A: List[int], K: int) -> List[int]:
        return list(str(int(''.join(map(str, A))) + K))
```

#### 628.三个数的最大乘积
1. 排序法
	1. 算法思路
		1. 将数组排序
		2. 如果全部为正数或负数，则最大的三个数的乘积为最大值a
		3. 如果有两个或以上的负数，则最小的两个数和最大的数的乘积为最大值b
		4. 返回max(a,b)即可
	2. 复杂度分析
		1. 时间复杂度：O(nlogn)， 排序需要nlogn的时间
		2. 空间复杂度：O(n)，排序用了数组n的空间
```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        # 排序
        nums.sort()
        a = nums[-1] * nums[-2] * nums[-3]  # 最大的三个数的乘积
        b = nums[0] * nums[1] * nums[-1]  # 最小的两个数和最大的数的乘积
        return max(a, b)
```

2. 三数排序
	1. 寻找数组中三个最大值，和两个最小值。
	2. 最大乘积的返回值，仍然是最大三个数的乘积，与最小两个数和最大数的乘积之间比较，返回较大者。
	3. 复杂度分析
		1. 时间复杂度：O(n)，需要遍历一次数组
		2. 空间复杂度：O(n)，需要保存两个中间数组
```python
class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        min1, min2 = float('inf'), float('inf')
        max1, max2, max3 = float('-inf'), float('-inf'), float('-inf')
        for num in nums:
            min1, min2, _ = sorted([min1, min2, num])
            _, max1, max2, max3 = sorted([max1, max2, max3, num])
        return max(min1 * min2 * max3, max1 * max2 * max3)
```

#### 127.单词接龙
1. 广度优先搜索（BSF）
	1. 将[wordList]()变为[set]()，方便后续匹配符合的单词查询
	2. 考虑[endWord]()不在[wordList]()里的特例，返回0
	3. 建立[queue]()，先将[beginWord]()放入队列，进行bfs搜索
		1. 如果弹出的单词[curWord]()==[endWord]()，则返回计数[step]()
		2. 如果弹出的单词[curWord!=endWord]()，则对[curWord]()进行单词变换处理，将每一位字母用小写26个字母进行替换构成[tmp]()，如果[tmp]()在[wordList]()匹配有结果且没有访问过，就将[tmp]()加入队列[queue]()和访问[visted]()，计数[step][19]+1。
	4. 复杂度分析
		1. 时间复杂度：O(m\*n)，m为[wordList]()中个数，n为单词长度
		2. 空间复杂度：O(n)，queue和set都需要额外空间
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        from collections import deque
        # wordList用set处理
        st = set(wordList)
        # 处理特例
        if endWord not in st:
            return 0
        # 建立队列进行bfs
        queue = deque()
        queue.append([beginWord,1])
        visted = set()
        visted.add(beginWord)
        m=len(beginWord)
        while queue:
            cur, step = queue.popleft()
            # 当当前单词和目标单词一致时，返回
            if cur == endWord:
                return step
            # 当当前单词和目标单词不一致时，搜索wordlist中的可以匹配的单词
            for i in range(m):
                for j in range(26):
                    tmp = cur[:i] + chr(97 + j) + cur[i + 1:]
                    if tmp in st and tmp not in visted:
                        queue.append([tmp, step + 1])
                        visted.add(tmp)

        return 0
```

#### 367.有效的完全平方数
数字[num]()是一个有效的完全平方数，则有[x\*x==num]()
1. 二分查找
	1. 如果[n\<2]()时，返回[true]()（特例）
	2. 设置左边界[left==2]()，右边界[right==num/2]()，当[left\<=right]()时，令[x=(left+right)/2]()，计算[tmp=x\*x]()，判断x与num的结果：
		1. 当[tmp==num]()时，返回[true]()
		2. 当[tmp\>num]()时，令[right=x-1]()
		3. 当[tmp\<num]()时，令[left=x+1]()
	3. 如果在[left\<=right]()里没有找到，返回[false]()
	4. 复杂度分析
		1. 时间复杂度：O(logN)
		2. 空间复杂度：O(1)
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # 特例
        if num < 2:
            return True

        left, right = 2, num // 2  # //整除

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
	1. 牛顿迭代法公式：[x=1/2(x+num/x)]()
	2. 根据公式，取[num/2]()作为初始近似解；当[x\*x\>num]()时不停迭代公式，直到循环终止。
	3. 返回[x\*x==num]()的结果
	4. 复杂度分析
		1. 时间复杂度：O(logN)
		2. 空间复杂度：O(1)
```python
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        # 特例
        if num < 2:
            return True
        # 牛顿迭代法
        x = num // 2
        while x * x > num:
            x = (x + num // x) // 2
        return x * x == num
```

#### 33.搜索旋转排序数组
1. 二分查找
	1. 设置左边界[left=0][42]、右边界[right=len(num)-1]()和中间[mid=(left+right)/2]()
	2. 当[left\<=right]()时，有三种情况判断
		1. 当[num\[mid]==target]()是，返回[mid]()
		2. 当[num[0]\<num\[mid]]()时，即[num\[0]]()至[num\[mid]]()是有序数组，这时又有两种情况：
			1. 当[num\[0]\<=target\<num\[mid]]()时，意味目标在有序数组中，移动右边界[right=mid-1]()寻找目标
			2. 否则目标在数组另一部分中，移动左边界[left=mid+1]()寻找
		3. 同理，当[num\[0]\>num\[mid]]()时，即[num\[mid]]()至[num\[-1]]()是有序数组，同理又有两种情况
			1. 当[num\[mid]\<target\<=num\[-1]]()时，意味目标在有序数组中，移动左边界[left=mid+1]()寻找目标
			2. 否则目标在数组另一部分中，移动右边界[right=mid+1]()寻找
	3. 复杂度分析
		1. 时间复杂度：O(logN)，二分查找的时间复杂度
		2. 空间复杂度：O(1)，常量变量
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if not nums:
            return -1
        # 初始化
        n = len(nums)
        left, right = 0, n - 1
        # 二分查找
        while left <= right:
            mid = (left + right) // 2
            # print(left, right, mid)
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[mid]:
                if nums[0] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[n - 1]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
```

#### 46.全排列（本周复习）
1. 分治递归
	1. 主体思路：当锚定nums[i]这个元素时，获取nums数组中除这个元素外的其他元素的全排列组合。
	2. 将1点提炼成重复子问题递，则有[premute(nums\[:i]+nums\[i+1:])]()的递归工作
	3. 不断分解2点，则有[for p in premute(nums\[:i]+nums\[i+1:])]()的循环
	4. 最终返回[nums\[i]+p]()
	5. 复杂度分析
		1. 时间复杂度：O(N\*N!)：nums数组的长度n，和递归循环的n
		2. 空间复杂度：O(N\*N!)：n为递归生成的系统栈
```python
def permute(self, nums: List[int]) -> List[List[int]]:
        if not nums: return [[]]

        res = []
        for i in range(len(nums)):
            for p in self.permute(nums[:i] + nums[i+1:]):
                res += [[nums[i]] + p]
        return res
```

2. 一份国际版参考代码
	利用了库函数Counter
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(counter, path):
            if len(path) == len(nums):
                res.append(path)
                return
            for x in counter:
                if counter[x]:
                    counter[x] -= 1
                    dfs(counter, path+[x])
                    counter[x] += 1
        dfs(Counter(nums), [])
        return res 
```

#### 47.全排列II（本周复习）
1. 回溯法
	1. 因为序列中右重复元素，所以会产生重复的排列，为避免重复结果的产生，需要做剪枝处理，为方便做剪枝，首先需要对序列进行排序。
	2. 当不是第一个元素，且当前元素与上个元素一致，且上个元素已被使用但又被重置为未被使用的状态时，进行剪枝处理避免重复结果当产生，则有条件[if i\>0 and nums\[i]==nums\[i-1] and visited\[i-1] is not False then:  continue]()
	3. 复杂度分析
		1. 时间复杂度：O(N\*N!)
		2. 空间复杂度：O(N\*N!)
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        visited = [False for _ in range(len(nums))]
        # print(visited)
        # 排序重要
        nums.sort()
        self.dfs(nums, visited, [], res)
        return res

    def dfs(self, nums, visited, path, res):
        if len(nums) == len(path):
            res.append(path.copy())
            return
        for i in range(len(nums)):
            if visited[i] is False:
				#剪枝处理
                if i > 0 and nums[i] == nums[i - 1] and not visited[i - 1]:
                    continue
                visited[i] = True
                self.dfs(nums, visited, path + [nums[i]], res)
                visited[i] = False
```

2. 一份国际版参考代码
	仍然利用了库函数Counter，因为Counter函数的特点，所以全排列[没有重复数字的序列]()和全排列II[有重复数字的序列]()的差异点“消失”了。
```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(counter, path):
            if len(path) == len(nums):
                res.append(path)
                return
            for x in counter:
                if counter[x]:
                    counter[x] -= 1
                    dfs(counter, path+[x])
                    counter[x] += 1
        dfs(Counter(nums), [])
        return res
```

#### 169.多数元素
1. 利用字典（hash）
	1. 利用字典的[key, value]()特性，令元素=key，元素出现的次数=value。
	2. 遍历数组，判断[nums\[i]]()是否存在在字典[dict]()里：
		1. 当存在时，[dict\[key]+=1]()
		2. 不存在时，[dict\[key]=1]()
	3. 遍历字典dict，当[value\>len(num)/2]()时，返回字典的[key]()
	4. 复杂度分析
		1. 时间复杂度：O(N)，n为数组的长度，至少遍历一次
		2. 空间复杂度：O(N)，利用了额外空间保存字典
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        dict = {}
        #遍历数组，并对元素计数创建对应的字典
        for num in nums:
            if not dict.get(num):
                dict[num] = 1
            else:
                dict[num] += 1
        # print(dict)
        tmp = len(nums) // 2
        # print(tmp)
        for k, v in dict.items():
            # print(k , v )
            if v > tmp:
                return k
```

2. 摩尔投票法
	1. 算法的每一步都分两个阶段：
		1. 对抗阶段：分属两个候选人的票数进行两两对空抵消
		2. 计数阶段：计算对抗结果中留下的候选人的票数是否有效
	2. 假设候选人[major]()和票数[count]()，具体实施过程为在遍历下一个选票时，判断当前[count]()是否有效。
		1. 如果[count==0]()，则令当前[nums\[i]==major]()，且[count==1]()
		2. 如果[count!=0]()，则[count-=1]()，并继续
	3. 复杂度分析
		1. 时间复杂度：O(N)，n为数组长度
		2. 空间复杂度：O(1)，变量为常量
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 摩尔投票法
        major = nums[0]
        count = 1

        for i in range(1, len(nums)):
            if count == 0:
                major = nums[i]
                count = 1
            else:
                if nums[i] == major:
                    count += 1
                else:
                    count -= 1
        return major
```

3. 排序
	按题意，出现次数最多的元素肯定超过数组长度一半的以上，那么排序后取处于数组中位数的元素即可。
```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
		# pythonic
        return sorted(nums)[len(nums)//2]
```

#### 77.组合（本周复习）
1. 回溯法（递归）
	1. 将n视为1-n的数组，然后穷举长度为k的不重复的组合。这里的穷举工作即递归工作，因前面使用过的元素在后续组合排列中仍需使用，所以此点需要回溯。
		![]()
	2. 递归工作
		1. 终止条件：当中间结果[len(cur)==k]()时，即可将结果保存至最终结果[res]()中
		2. 当[len(cur)!=k]()时，则不断的将下一个元素**i**加入到**cur**中，调用递归时将元素下标**+1**。
		3. 递归结束后，进行[cur.pop()]()，进行回溯，以便后续组合使用
	3. 复杂度分析
		1. 时间复杂度：O(N)，遍历n
		2. 空间复杂度：O(N)，递归时调用系统栈，栈为K大小
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 定义dfs
        def backtrace(cur, idx):
            # 终止条件
            if len(cur) == k:
                res.append(cur[:])
                return
                # 逻辑处理与下钻
            for i in range(idx, n + 1):
                cur.append(i)
                backtrace(cur, i + 1)  # 下标+1
                cur.pop()  # 回溯

        res = []
        if k == 0:
            return res
        backtrace([], 1)
        return res
```

2. pythonic风格
```python
#国际版
class Solution:
    def combine(self, n, k):
        if k == 0:
            return [[]]
        return [pre + [i] for i in range(k, n+1) for pre in self.combine(i-1, k-1)]

#老师还原版本
def combine(self, n, k):
        if k == 0:
            return [[]]
        res = []
        for i in range(k, n+1):
            for pre in self.combine(i-1, k-1):
                res += [pre + [i]]
        return res
```

#### 153.寻找旋转排序数组中的最小值
1. 二分查找
	1. 设置左边界[left=0]()和右边界[right=len(nums)-1]()
	2. 当[left\<=right]()时，令[mid==(left+right)//2]()
		1. 当[nums\[mid]\<nums\[0]]()时，在mid左边搜索最小值
		2. 当[nums\[mid]\>nums\[0]]()时，在mid右边搜索最小值
			1. 如果有[nums\[mid]\>nums\[mid+1]]()，mid+1是最小值
			2. 如果右[nums\[mid-1]\>nums\[mid]]()，mid是最小值
	3. 复杂度分析
		1. 时间复杂度：O(logN)
		2. 空间复杂度：O(1)
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        left, right = 0, len(nums) - 1
        if nums[right] > nums[left]:
            return nums[0]
        while left < right:
            mid = (left + right) // 2
            print(nums[left], nums[mid], nums[right])
            if nums[mid] > nums[mid + 1]:
                return nums[mid + 1]
            elif nums[mid - 1] > nums[mid]:
                return nums[mid]
            elif nums[mid] > nums[0]:
                left = mid + 1
            else:
                right = mid - 1
```

2. 国际版参考代码
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = low + (high - low) // 2
            ele = nums[mid]
            if ele > nums[high]:
                low = mid + 1
            elif mid == 0 or nums[mid - 1] > nums[mid]:
                return nums[mid]
            else:
                high = mid - 1
```



[19]:	+
[42]:	%5C


