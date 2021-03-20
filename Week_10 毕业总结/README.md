# WEEK10学习笔记
### 毕业总结
#### **毕业感言**
不知不觉算法训练营已陪伴我们走过10周，即将结束。
首先，感谢覃超老师的悉心教导，助教郑敏的热心解答，班班的后勤工作，以及一起参与学习的各位小伙伴们的每日坚持。正因为大家汇聚一堂有共同的目标，才使显得枯燥无味的学习过程，变得有激励、有自律，让整个学习过程充满了热烈的讨论，70天内众人的努力和坚持是看得见和摸得着的。按OKR衡量，我们有关键目标，有关键过程，有可衡量的价值产出。70天的努力，让我们面对算法题目时，不再是曾经的无可作为，而是如今的兴趣盎然，勇于尝试。
#### **课程收获**
70天算法训练营的学习，收获十分丰富，回想下来，大致可归纳为以下几点：
##### **数据结构与算法**
数据结构、初级算法和高级算法，是训练营所教授的核心部分，通过系统化的学习，再次强化了对这些知识点的理解和记忆。初略统计有以下内容：
- 数据结构：数组、链表、队列、栈、堆、树、二叉树、字典树和并查集等
- 初级算法：迭代、递归、分治、回溯、深度优先搜索和广度优先搜索、贪心和动态规划等
- 高级算法：剪枝、高级动态规划、启发式算法，位运算、各类排序、布隆过滤器和LRU缓存等
- 另外在讲述这些知识点时，还有一些额外的用于启发兴趣的内容。
##### **学习方法（五毒神掌）**
五毒神掌的核心思想是过遍数。总结一下就是：
- 第一步，临摹、理解和学习前人的思想和方法，即在理解题解思路之后，照葫芦画瓢写出题解代码。
- 第二步，消化、吸收，将前人的算法思路变为自己的东西，可以根据自我理解写出答案。
- 第三步，反复练习，加深理解和记忆，沉淀出自己的知识点。
- 第四步，不断循环第一至三步，最终形成自己的知识树。
##### **坚持不懈的训练**
70天每日一题的刻意练习，培养出了坚持算法学习的自律行为。相信只要坚持不懈，这个刻意训练一定会带来更多的好处。同理，也可以通过同样的方法培养出其他方面的自律行为，对未来带来更多的价值意义。
##### **逻辑思维与抽象能力**
通过算法训练营这段时间的学习，经过不断的理解题目和解答题目，感觉对编程的逻辑思维能力和对事物的抽象能力均有了一定的提升。

### 本周leetcode练习总结
#### 83.删除排序链表中的重复元素
1. 直接法（单指针）
	1. 算法思路
		1. 因为链表已经排序，所以挨个比较相邻节点的值是否相等即可。如果相等，则将当前节点指向后继节点的后继节点，达到删除后继节点的目的。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次链表
		2. 空间复杂度：O(1)
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        dummy = head
        while dummy and dummy.next:
            if dummy.val==dummy.next.val:
                dummy.next=dummy.next.next
            else:
                dummy=dummy.next
        return head
```

2. 递归法
	1. 算法思路
		1. 链表具有天然递归性，一个链表可以看作是头节点后挂接一个更短的链表。
		2. 递归工作：判断头节点是否和头节点的后继节点是否相等，相等则返回后继节点的后继节点，否则返回头节点。
		3. 递归终止条件：头节点为空或头节点的后继节点为空。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        # 终止条件
        if not head or not head.next:
            return head
        head.next = self.deleteDuplicates(head.next)
        return head.next if head.val == head.next.val else head
```

#### 746.使用最小花费爬楼梯
1. 动态规划
	1. 算法思路
		1. 阶梯对应的下标为[0,n)，所以楼层顶的下标为n，求到n的最小花费即可。
		2. 创建dp数组，长度n+1；因为初始可以从0或1的下标开始，所以初始dp[0]=dp[1]=0即可。
		3. 对于dp[i]，其最小花费既有可能是dp[i-1]的状态值+cost[i-1]的值之和，也有可能是dp[i-2]的状态值+cost[i-2]的值之和。所以递推公式为：dp[i]=min(dp[i-1]+cost[i-1],dp[i-2]+cost[i-2])。
		4. 最终返回dp[-1]即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次cost数组
		2. 空间复杂度：O(N)，dp数组大小等于cost数组+1
	3. 题解代码
```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        dp = [0] * (n + 1)
        dp[0] = dp[1] = 0
        for i in range(2, n + 1):
            dp[i] = min(dp[i - 1] + cost[i - 1], dp[i - 2] + cost[i - 2])
        # print(dp)
        return dp[-1]
```
4. 空间复杂度优化
	1. 利用滚动数组思想优化空间复杂度
		2. 因为dp[i]的状态值，只和dp[i-1]和dp[i-2]有关。
		3. 优化后的空间复杂度：O(1)
		4. 题解代码
```python
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        n = len(cost)
        pre = cur = 0
        for i in range(2, n + 1):
            nex = min(cur + cost[i - 1], pre + cost[i - 2])
            pre, cur = cur, nex
        return cur
```

#### 77.组合（本周复习）
1. 回溯法
	1. 算法思路
		1. n表示1-n的数，k表示组合长度，要求穷举所有组合。因为使用过的元素再后续组合还将继续使用，所以在递归工作中要进行回溯。
		2. 递归工作具体表现
			1. 递归终止条件，当组合结果tmp的长度==k时，将tmp加入到最终结果中，并返回即可。
			2. 当不符合终止条件时，遍历1-n的数字，变量为i，将数字i加入组合结果tmp中；然后调用自身函数进行递归，传入tmp和i+1；递归后将i从tmp移出，以便后续使用。
	2. 复杂度分析
		1. 时间复杂度：O(N\*K)，N为1-n的遍历，k为递归调用次数
		2. 空间复杂度：O(N)，临时空间为递归的系统栈加结果存储空间
	3. 题解代码
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        # 回溯
        def dfs(tmp, idx):
            # 递归终止
            if len(tmp) == k:
                res.append(tmp[:])
                return
                # 处理当前值
            for i in range(idx, n + 1):
                tmp.append(i)
                # print(['a']+tmp)
                dfs(tmp, i + 1)
                tmp.pop()  # 回溯
                # print(['b']+tmp)

        # 主逻辑
        res = []
        if k == 0:
            return res
        dfs([], 1)
        return res
```

#### 120.三角形最小路径和
1. 动态规划（自上而下）
	1. 算法思路
		1. 将三角形的左边对齐后，即是一个等腰三角形。
		2. 定义一个dp数组，等同于等腰三角形大小，这里有三种情况考虑
			1. 一般情况：dp[i][j]的最小路径和等于dp[i-1][j-1]和dp[i-1][j]中的较小者+triangle[i][j]的值，即有公式：dp[i][j]=min(dp[i-1][j-1],dp[i-1][j])+triangle[i][j]。
				![]()
			2. 左边界，即三角形的第一列，状态值的转移只能从上至下：dp[i][0]的最小路径和等于dp[i-1][0]+triangle[i][0]的和，即：dp[i][0]=dp[i-1][0]+triangle[i][0]。
				![]()
			3. 三角形的斜边，即状态值的转移只能从左上至右下：dp[i][j]的最小路径和等于dp[i-1][j-1]+triangle[i][j]的和，即：dp[i][j]=dp[i-1][j-1]+triangle[i][j]。
				![]()
		3. 最终返回dp数组最后一行的最小值即可。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)
		2. 空间复杂度：O(N^2)
	3. 题解代码
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        # 初始dp
        n = len(triangle)  # 三角形的行
        m = len(triangle[-1])  # 三角形的最后一行的列，最长
        dp = [[0] * m for _ in range(n)]
        dp[0][0] = triangle[0][0]
        # dp数组
        for i in range(1, n):
            # dp的三角形第一列位置赋值
            dp[i][0] = dp[i - 1][0] + triangle[i][0]
            j = 1
            # 不包含斜边
            while j < len(triangle[i]) - 1:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j]) + triangle[i][j]
                j += 1
            # 斜边位置赋值
            dp[i][j] = dp[i - 1][j - 1] + triangle[i][j]
        # 返回dp最后一行的最小值
        # print(dp)
        return min(dp[-1])
```

2. 动态规划（自下而上）
	1. 算法思路
		1. 与自上而下相比，dp数组的不同点是
			1. dp数组的原始行数+1，主要为了处理边界条件
			2. 最终的最小路径和保存在dp[0][0]的位置。
		2. 状态转移公式：dp[i][j]=min(dp[i+1][j+1],dp[i+1][j])+triangle[i][j]。
			![]()
		3. 遍历dp数组的过程是逆序遍历（i：从下到上，j：从右到左）。
		4. 最终返回dp[0][0]即可。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)
		2. 空间复杂度：O(N^2)
	3. 题解代码
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        # 初始dp
        n = len(triangle)  # 三角形的行
        m = len(triangle[-1])  # 三角形的最后一行的列，最长
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(len(triangle[i]) - 1, -1, -1):
                dp[i][j] = min(dp[i + 1][j + 1], dp[i + 1][j]) + triangle[i][j]
        # print(dp)
        return dp[0][0]
```
4. 空间压缩优化
	1. 因为计算dp[i][j]时，只需要dp[i+1]这一行的状态值即可，所以可以进行空间压缩优化。将dp数组变为一维数组。
		2. 状态转移公式：dp[j]=min(dp[j],dp[j+1])+triangle[i][j]。
			![]()
		3. 最终返回dp[0]即可。
		4. 复杂度分析
			1. 时间复杂度：O(N^2)
			2. 空间复杂度：O(N)
		5. 题解代码
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        # 初始dp
        n = len(triangle)  # 三角形的行
        m = len(triangle[-1])  # 三角形的最后一行的列，最长
        dp = [0] * (m + 1)
        for i in range(n - 1, -1, -1):
            # j顺序遍历
            for j in range(len(triangle[i])):
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
        # print(dp)
        return dp[0]
```

#### 86.分隔链表（本周复习）
1. 拼接链表
	1. 算法思路
		1. 创建两个空链表p和q。
		2. 原始链表中小于x的节点，按原顺序加入p。
		3. 原始链表中大于x的节点，按原顺序加入q。
		4. 最后返回p-\>q即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次原始链表
		2. 空间复杂度：O(N)，p+q的空间等于原始链表空间
	3. 题解代码
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        # 初始两个链表
        p = less = ListNode(0)
        q = more = ListNode(0)
        # 遍历head
        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                more.next = head
                more = more.next
            head = head.next
        # 拼接p和q
        more.next = None
        # print(less)
        # print(more)
        # print(p)
        # print(q)
        less.next = q.next
        return p.next
```

1047.删除字符串中的所有相邻重复项
1. 利用栈
	1. 算法思路
		1. 利用栈，逐个遍历字符串的元素。
		2. 当当前元素和栈顶元素一致时，则相互抵消；否则压入栈。
		3. 最终返回字符串化的栈即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次字符串
		2. 空间复杂度：O(N)，视栈利用的空间
	3. 题解代码
```python
class Solution:
    def removeDuplicates(self, S: str) -> str:
        # 栈
        stack = []
        for c in S:
            if stack and c == stack[-1]:
                stack.pop()
            else:
                stack.append(c)
        return ''.join(stack)
```

#### 127.单词接龙（本周复习）
1. 单向BFS
	1. 算法思路
		1. 将wordList转为set集合，降低后续判断变量单词tmp是否在单词表中的时间复杂度。
		2. 初始判断endWord是否在wordList中，不存在直接返回0。
		3. 初始建立队列queue，将beginWord和1作为初始元组加进去。同时建立visited集合，将beginWord添加进去。
		4. 当queue不为空时循环进行以下操作：
			1. queue弹出首端元素，赋值cur和step，分别是单词和次数。
			2. 当cur==endWord时，表示接龙成功，返回step即可。
			3. 否则遍历cur单词的每个字母，使用26个字母进行迭代替换生成新单词tmp。
			4. 新单词tmp，如果未被访问过且在wordList中时，将tmp和step+1加入队列queue，同时tmp也添加至visited。
		5. 当queue遍历完毕后，仍未找到，返回0。
	2. 复杂度分析
		1. 时间复杂度：O(N\*M)，N为wordList的单词个数，M为单个单词长度。
		2. 空间复杂度：O(N)，queue、wordList和visited开辟的内存空间。
	3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        from collections import deque
        st = set(wordList)
        if endWord not in st:
            return 0
        # 初始化
        queue = deque()
        queue.append((beginWord, 1))
        visited = set(beginWord)
        # 遍历queue
        while queue:
            cur, step = queue.popleft()
            if cur == endWord:
                return step
            else:
                # 变换单词
                for i in range(len(cur)):
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        tmp = cur[:i] + j + cur[i + 1:]
                        # print(tmp)
                        if tmp in st and tmp not in visited:
                            queue.append((tmp, step + 1))
                            visited.add(tmp)
        #
        return 0
```

2. 双向BFS
	1. 算法思路
		1. 总体思路与单向BFS一致，不同点在于以下：
			1. 需要分别为beginWord和endWord建立队列和访问集合，如leftqueue、rightqueue、leftvisited和rightvisited。
			2. 接龙次数的计算以层计。
			3. **判断两个队列的长度**，每次从**较短**的队列中取出元素，与另一侧的visited集合中单词做匹配，如匹配成功则表示接龙成功。
		2. 复杂度分析
			1. 时间复杂度：O(N\*M)，与单向BFS计算方式一致，但比单向BFS略快。
			2. 空间复杂度：O(N)
		3. 题解代码
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        from collections import deque
        st = set(wordList)
        if endWord not in st:
            return 0
        # 初始化
        leftqueue = deque()
        leftqueue.append(beginWord)
        leftvisited = set()
        leftvisited.add(beginWord)
        rightqueue = deque()
        rightqueue.append(endWord)
        rightvisited = set()
        rightvisited.add(endWord)
        step = 0
        # 遍历queue
        while leftqueue and rightqueue:
            step += 1  # 循环一次，层级+1
            # print(step)
            # 找出较短的队列
            if len(leftqueue) > len(rightqueue):
                leftqueue, leftvisited, rightqueue, rightvisited = rightqueue, rightvisited, leftqueue, leftvisited
            # 对当前队列继续处理
            for k in range(len(leftqueue)):
                cur = leftqueue.popleft()
                # print(cur)
                if cur in rightvisited:
                    return step
                else:
                    # 变换单词
                    for i in range(len(cur)):
                        for j in 'abcdefghijklmnopqrstuvwxyz':
                            tmp = cur[:i] + j + cur[i + 1:]
                            # print(tmp)
                            if tmp in st and tmp not in leftvisited:
                                leftqueue.append(tmp)
                                leftvisited.add(tmp)
        # 没有结果返回0
        return 0
```

#### 88.合并两个有序数组（本周复习）
1. 三指针
	1. 算法思路
		1. 利用从右往左的顺序，将两个数组中的较大值逐步从nums1的逆序开始放置。
		2. 设置i=m-1、j=n-1和k=m+n-1，进行以下操作：
			1. 当nums1[i]\>nums2[j]时，nums1[k]=nums1[i]，k-=1和i-=1。
			2. 反之亦然
		3. 当第2点完成后，如果j\>=0，表示nums2还有剩余，再将nums2剩余元素填补到nums1中。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次nums1长度和nums2剩余长度，非嵌套。
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i, j, k = m - 1, n - 1, m + n - 1
        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                k -= 1
                i -= 1
            else:
                nums1[k] = nums2[j]
                k -= 1
                j -= 1
        # 处理剩余的nums2
        if j >= 0:
            while j >= 0:
                nums1[k] = nums2[j]
                k -= 1
                j -= 1
```

2. 切片+排序
	1. 算法思路
		1. 将nums2通过切片拼接到nums1后面
		2. 将nums1排序
	2. 复杂度分析
		1. 时间复杂度：O(NlogN)，排序耗时
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[:] = nums1[:m] + nums2
        nums1.sort()
```

#### 46.全排列（本周复习）
1. 回溯法
	1. 算法思路
		1. 利用DFS遍历树的所有节点，对所有节点增加使用状态标识。向下递归的时候，节点打上“使用”标识；向上回溯的时候，对已使用的节点打上“撤销”标识，以便在后续组合中继续使用。
		2. 递归工作：
			1. 建立used数组，用于保存所有节点的使用状态，初始为false，当使用过后标为true。
			2. 终止条件：设定向下层次变量depth，当depth==nums长度时，表示排列的长度已经足够，可以加入到结果res中。
			3. 对未使用过的节点进行进行以下操作
				1. 将节点的used状态标为true，表示已使用。
				2. 将节点加入中间结果tmp中
				3. 继续递归工作，递归下一层时depth+1，同时也将中间结果tmp和状态used一并传入。
			4. 递归完成后进行回溯，将使用过的节点的状态标为false，以便后续使用；同时从中间结果tmp中删除已使用的节点。
	2. 复杂度分析：
		1. 时间复杂度：O(N\*N!)，遍历数组为N，每个内部节点循环为N，非叶子节点时间为O(N\*N!)；最后一层有N!个叶子节点，结果拷贝需要O(N)，时间复杂度同为O(N\*N!)。最终时间复杂度O(N\*N!)。
		2. 空间复杂度：O(N\*N!)，递归深度logN，全排列个数N！，每个排列为N，最终为O(N\*N!)。
	3. 题解代码
```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        # uesd数组，记录节点状态
        used = [False] * n
        res = []

        # 回溯
        def dfs(depth, tmp, used):
            # 递归终止：当depth==n时，表示排列完成
            if depth == n:
                res.append(tmp.copy())
                return
                # 遍历节点
            for i in range(n):
                # 未使用的节点处理
                if not used[i]:
                    used[i] = True
                    tmp.append(nums[i])
                    # 递归工作
                    dfs(depth + 1, tmp, used)
                    # 状态重置
                    used[i] = False
                    tmp.pop()

        # 从第一层开始递归
        dfs(0, [], used)
        return res
```

#### 94.二叉树的中序遍历（本周复习）
1. 递归法
	1. 算法思路
		1. 二叉树的中序遍历的顺序为：左-\>根-\>右
		2. 当root为空的时候，返回空；否则将root的值加入到结果中。
		3. 按中序遍历顺序进行递归操作，返回依次递归左节点+根节点的值+递归右节点的结果即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历二叉树的所有节点
		2. 空间复杂度：O(N)，递归调用的系统栈
	3. 题解代码
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 初始化
        res = []
        if not root:
            return res
        # 中序：左-根-右
        return self.inorderTraversal(root.left) + [root.val] + self.inorderTraversal(root.right)
```

#### 709.转换成小写字母
1. 线性扫描
	1. 算法思路
		1. 遍历字符串str每个字母，利用ord函数将大写字母改变为小写字母。大小写字母的ascii码差32位。
	2. 复杂度分析
		1. 时间复杂度：O(N)，字符串长度
		2. 空间复杂度：O(N)，结果res的空间，与字符串长度一致。
	3. 题解代码
```python
class Solution:
    def toLowerCase(self, str: str) -> str:
        res = ''
        for c in str:
            if c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                c = chr(ord(c) + 32)
            else:
                pass
            res += c
        return res
```

#### 52.N皇后II
1. 回溯+剪枝
	1. 算法思路
		1. 思路与51.N皇后的解法一致，不同在于，需要求解N皇后问题不同解决方案的数量，而非打印棋盘的摆放位置。
		2. 将N皇后的题解代码稍做修改即可，将res改为数字结果，每当找到一个可能解时，res+1。这里需要注意的是res要设为全局变量。
	2. 复杂度分析
		1. 时间复杂度：O(N!)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def totalNQueens(self, n: int) -> int:
        # 回溯
        def dfs(queens, xy_diff, xy_sum):
            nonlocal res
            p = len(queens)  # 算行数，行列相等
            if p == n:
                res += 1
                return
            # 遍历每列
            for q in range(n):
                # 剪枝 ，逆向思维，能放Q的位置必须同时满足不在列、撇、捺里
                if q not in queens and p - q not in xy_diff and p + q not in xy_sum:
                    # 递归操作
                    dfs(queens + [q], xy_diff + [p - q], xy_sum + [p + q])

        res = 0
        dfs([], [], [])
        return res
```

#### 122.买卖股票的最佳时机II（本周复习）
1. 动态规划
	1. 算法思路
		1. 根据题目意思，在第i天时，通过买卖股票获取的最大利润dp[i]可能存在两种情况：一是持有股票时的最大利润，二是未持有股票时的最大利润。故动态递推考虑以下场景：
			1. 第i天持有股票的最大利润，来自于第i-1天已经持有股票时的利润，或者第i-1天未持有股票但在第i天购买了股票后的利润。故状态转移方程：dp[i][1]=max(dp[i-1][1],dp[i-1][0]-price[i])。
			2. 第i天未持有股票的最大利润，来自于第i-1天未持有股票时的利润，或者第i-1天持有股票但在第i天卖出股票后的利润。故状态转移方程：dp[i][0]=max(dp[i-1][0],dp[i-1][1]+price[i])。
		2. 全部交易结束后，持有股票的收益肯定小于未持有股票的收益，所以最终的最大利润在dp[i][0]中。
		3. 初始dp数组时，dp[0][0]=0，dp[0][1]=-price[0]（第一天购买了股票。）
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次price数组
		2. 空间复杂度：O(N)，dp数组所需空间
	3. 题解代码
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        # 初始dp数组
        dp = [[0] * 2 for _ in range(len(prices))]
        # 第一天初始
        dp[0][0] = 0  # 未持有
        dp[0][1] = -prices[0]  # 持有
        # print(dp)
        # 遍历交易
        for i in range(1, len(prices)):
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        # print(dp)
        return dp[-1][0]
```

#### 102.二叉树的层序遍历（本周复习）
1. 递归（DFS）
	1. 算法思路
		1. 通过递归工作将每层遍历到节点保存至对应的层级结果中。
		2. 具体递归工作
			1. 终止条件：当节点为空时，终止返回。
			2. 节点不为空时，将节点添加至对应的层级结果中。当遍历到一个新的层级时，要在结果中创建新数组用于保存结果。
			3. 对节点的左节点和右节点进行递归调用，层级相应+1。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历树的所有节点
		2. 空间复杂度：O(N)
	3. 题解代码
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # 递归
    def dfs(self, root, level, res):
        # 终止
        if not root:
            return
            # 遇到新层级时，在结果创建一个数组
        if len(res) == level:
            res.append([])
        # 节点加入level的数组中
        res[level].append(root.val)
        # 递归调用左右子节点
        self.dfs(root.left, level + 1, res)
        self.dfs(root.right, level + 1, res)

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        self.dfs(root, 0, res)
        return res
```

#### 238.除自身以外数组的乘积
1. 左右乘积法
	1. 算法思路
		1. 数组nums的下标i，除i代表的元素外的其他数的乘积，等同于i左边元素的乘积\*i右边元素的乘积。
		2. 初始化L和R两个数组，根据下标i，L[i]代表i左侧所有数字的乘积，R[i]代表i右侧所有数字的乘积。
		3. 数组L，顺序遍历数组，令L[i]=nums[i-1]\*L[i-1]，i从1起，L[0]=1。
		4. 数组R，逆序遍历数组，令R[i]=nums[i+1]\*R[i+1]，i从n-1起，R[n-1]=1。
		5. 对于结果数组res，res[i]=L[i]\*R[i]
	2. 复杂度分析
		1. 时间复杂度：O(N)，对数组nums遍历了3次，但非嵌套。
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [0] * n
        L = [0] * n
        L[0] = 1
        R = [0] * n
        R[-1] = 1
        # 左侧乘积
        for i in range(1, n):
            L[i] = nums[i - 1] * L[i - 1]
        # 右侧乘积
        for i in range(n - 2, -1, -1):
            # print(nums[i+1],' ',R[i+1])
            R[i] = nums[i + 1] * R[i + 1]
        # print(L)
        # print(R)
        # 除i的乘积
        for i in range(n):
            res[i] = L[i] * R[i]
        return res
```
 
2. 乘积矩阵（上三角和下三角）
	1. 算法思路
		1. 将结果数组res列为乘积形式的矩阵，发现下标i所在位置正式矩阵主对角线，且i所在位置均为1。
			![]()
		2. 分别计算矩阵的下三角和上三角，通过两次遍历，并存储过程值，可得到最终结果。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(1)，不算输出数组，过程数值保存在常数变量中。
	3. 题解代码
```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        res, p, q = [1], 1, 1
        for i in range(len(nums) - 1):
            p *= nums[i]  # i左侧乘积
            res.append(p)  # 保存于res
        # print(res)
        for j in range(len(nums) - 1, 0, -1):
            q *= nums[j]  # i右侧乘积
            res[j - 1] *= q  # 更新res
        # print(res)
        return res
```

#### 104.二叉树的最大深度（本周复习）
1. 递归
	1. 算法思路
		1. 利用递归遍历跟节点的左右子树的深度，再两者之间选择较大者+跟节点的层次即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        # 没有节点时返回0
        if not root:
            return 0
        # 递归左右子树
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
```

#### 91.解码方法
1. 动态规划
	1. 算法思路
		1. 定义：dp[i]，以s[i]结尾的前缀字符串有多少种解码方法。
		2. 状态转移方程推导：
			1. 如果s[i]==‘0’，s[i]不能单独解码；当s[i]!=‘0’时，dp[i]=dp[i-1]。
			2. 如果s[i-1:i+1]可以解码，即10\<=s[i-1:i+1]\<=26，得到dp[i]=dp[i-2]+dp[i]
		3. 初始化：如果s[0]==‘0’，则不能解码，返回0；否则dp[0]=1。
		4. 最终返回dp[-1]。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def numDecodings(self, s: str) -> int:
        if not s or s[0] == '0':
            return 0
        n = len(s)
        dp = [0] * n
        dp[0] = 1
        # print(dp)
        for i in range(1, n):
            if s[i] != '0':
                dp[i] = dp[i - 1]
            if 10 <= int(s[i - 1:i + 1]) <= 26:
                if i == 1:
                    dp[i] += 1
                else:
                    dp[i] += dp[i - 2]
        # print(dp)
        return dp[-1]
```

#### 239.滑动窗口最大值（本周复习）
1. 单调队列
	1. 算法思路
		1. 设定队列win用于保存nums元素下标；队列res保存结果值。
		2. 一次遍历数组nums，考虑以下几个点：
			1. 当下标i\>=窗口大小k时，且队列win的首端元素\<=i-k，表示该首端元素移出了窗口大小，需要从队列win里移除。
			2. 当队列win不为空，且队列末端元素nums[win[-1]]\<nums[i]时，需要将队列win中小于nums[i]的元素下标删除，保持队列单调性。
			3. 不满足第1、2点的元素下标i加入队列win中
			4. 当i\>=k-1时，即窗口形成，这时将每次移动的窗口最大值加入到结果res中。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组nums
		2. 空间复杂度：O(N)，队列win所需空间。
	3. 题解代码
```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 初始两个队列
        win = []
        res = []
        # 遍历一次数组
        for i in range(len(nums)):
            # 当形成窗口后，且win[0]移出了窗口范围，win[0]删除
            if i >= k and win[0] <= i - k:
                win.pop(0)
            # 当win不为空，清空win中所有<i代表的元素的下标
            while win and nums[win[-1]] < nums[i]:
                win.pop()
            # 不满足以上，将i加入win
            win.append(i)
            # 当有窗口后，每次移动时将窗口的最大值加入res
            if i >= k - 1:
                res.append(nums[win[0]])
        # 返回结果
        return res
```

#### 105.从前序与中序遍历序列构造二叉树（本周复习）
1. 递归法
	1. 算法思路
		1. 根据二叉树前序顺序：根-\>左-\>右，和中序顺序：左-\>根-\>右。可以得知跟节点root在前序遍历中是第一位，推导出其在中序遍历中的位置，从而划分出左子树和右子树在前序和中序中的位置范围。
		2. 通过递归工作，根据前序遍历中的左子树和中序遍历中的左子树构造出左子树；同理构造右子树。
		3. 最终返回root即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 递归终止：没有左右子树返回
        if not preorder and not inorder:
            return
        # 构造根节点
        root = TreeNode(preorder[0])
        # 找出跟节点在中序中的位置
        mid_idx = inorder.index(preorder[0])
        # 递归自身构造左右子树
        root.left = self.buildTree(preorder[1:mid_idx + 1], inorder[:mid_idx])
        root.right = self.buildTree(preorder[mid_idx + 1:], inorder[mid_idx + 1:])
        # 返回root
        return root
```

#### 119.杨辉三角II（本周复习）
1. 动态规划
	1. 算法思路
		1. 创建二维数组dp[i][j]，根据杨辉三角特点，可以得到状态转移方程：dp[i][j]=dp[i-1][j-1]+dp[i-1][j]。
		2. 初始时，杨辉三角的两条边都要初始化为1。
		3. 最终返回dp[-1]，即最后一行即可。
	2. 复杂度分析
		1. 时间复杂度：O(N^2)，N=输入的k值
		2. 空间复杂度：O(k\*(k+1)/2)
	3. 题解代码
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [[1 for j in range(i + 1)] for i in range(rowIndex + 1)]
        # print(dp)
        for i in range(2, rowIndex + 1):
            for j in range(1, i):
                dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j]
        # print(dp)
        return dp[-1]				
```
4. 滚动数组空间优化
	1. 使用长度为k的一维数组。
		2. 从右往左遍历，每个元素等于其左边元素加上自身：dp[j]+=dp[j-1]。从左向右遍历会破坏i-1个元素的状态。
		3. 空间复杂度降低至O(k)。
		4. 题解代码
```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        dp = [1] * (rowIndex + 1)
        # print(dp)
        for i in range(2, rowIndex + 1):
            for j in range(i - 1, 0, -1):
                dp[j] += dp[j - 1]
        # print(dp)
        return dp
```

#### 120.三角形最小路径和（本周复习）
1. 动态规划（自底向上）
	1. 算法思路
		1. 由三角形最下一行往上进行递推操作，到达第i行j列的最小路径和的值，等同于第i+1行j列和第i+1行j+1列中的较小值+三角形第i行j列的值。可得递推公式：dp[i][j]=min(dp[i+1][j+1],dp[i+1][j])+triangle[i][j]。
		2. 如果对三角形的数组进行原地修改，则不需要额外的dp空间数组。
```python
原始三角形测试用例：[[2], [3, 4], [6, 5, 7], [4, 1, 8, 3]]
自下而上和原地修改：[[11], [9, 10], [7, 6, 10], [4, 1, 8, 3]]
```
2. 复杂度分析
	1. 时间复杂度：O(N\*M)，N为三角形的行数，M为每行的长度。
		2. 空间复杂度：O(1)，原地修改
	3. 题解代码
```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        if not triangle:
            return 0
        # 原地修改三角形
        # 倒序遍历，自下而上
        # print(triangle)
        for i in range(len(triangle) - 2, -1, -1):  # 逆序遍历每行
            for j in range(len(triangle[i])):  # 对每行进行遍历
                triangle[i][j] = min(triangle[i + 1][j + 1], triangle[i + 1][j]) + triangle[i][j]
        # print(triangle)
        return triangle[0][0]  # 最小路径和在三角形顶点
```

#### 121.买卖股票的最佳时机（本周复习）
1. 动态规划
	1. 算法思路
		1. 定义dp[i][0]为第i天时未持有股票的状态，dp[i][1]为第i天时持有股票的状态，那么在第i天时最大利润有两种可能：
			1. 第i天时未持有股票的最大利润有两种情况：可能为第i-1天时持有在第i天卖出；也可能为第i-1天未持有的情况在第i天保持未持有的状态。故状态转移方程: dp[i][0]=max(dp[i-1][1]+price[i],dp[i-1][0])。
			2. 第i天持有股票时的最大利润有两种情况：可能在第i天买入，只能买一次；也可能第i-1天持有股票在第i天保持不变。故状态转移方程：dp[i][1]=max(-price[i],dp[i-1][1])。
		2. 最终最大利润在dp[i][0]中，因为第i天未持有股票的利润肯定大于持有的情况。
	2. 复杂度分析
		1. 时间复杂度：O(N)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        # 初始化dp
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = 0    #未持有
        dp[0][1] = -prices[0]   #持有
        # print(dp)
        # 遍历prices
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][1] + prices[i], dp[i - 1][0])
            dp[i][1] = max(- prices[i], dp[i - 1][1])
        # print(dp)
        # 返回dp[i][0]
        return dp[-1][0]
```

#### 122.买卖股票的最佳时机II（本周复习）
1. 动态规划
	1. 算法思路
		1. 定义状态dp[i][j]表示第i天时获取的最大利润，其中dp[i][0]表示第i天时未持有股票的利润，dp[i][1]表示第i天时持有股票的利润。
		2. 第i天未持有股票的最大利润，可能来自两种情况：
			1. 第i-1天持有股票，在第i天卖出后的利润
			2. 第i-1天未持有股票，在第i天保持未持有的利润
			3. 得到状态转移方程：dp[i][0]=max(dp[i-1][1]+prices[i],dp[i-1][0])
		3. 第i天持有股票的最大利润，可能来自两种情况：
			1. 第i-1天未持有股票，在第i天买入后的剩余利润
			2. 第i-1天持有股票，在第i天保持不变的利润
			3. 得到状态转移方程：dp[i][1]=max(dp[i-1][0]-prices[i],dp[i-1][1])
		4. 因未持有股票时的利润肯定大于持有时的利润，最终返回dp[-1][0]即可。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        # 初始化dp
        n = len(prices)
        dp = [[0] * 2 for _ in range(n)]
        dp[0][0] = 0  # 第一天未持有
        dp[0][1] = -prices[0]  # 第一天买入
        # print(dp)
        for i in range(1, n):
            dp[i][0] = max(dp[i - 1][1] + prices[i], dp[i - 1][0])
            dp[i][1] = max(dp[i - 1][0] - prices[i], dp[i - 1][1])
        # print(dp)
        # 最大利润在未持有
        return dp[-1][0]
```

#### 1603.设计停车系统
1. 二维数组
	1. 算法思路
		1. 设计一个二维数组restcar[i][j]，其中restcar[i][0]表示每种车型可以停放的上限值，下标restcar[i][1]表示该车型已经停放了数目。
		2. 根据cartype的值，将相应的restcar[i][1]进行+1；当restcar[i][0]\>=restcar[i][1]时，返回true，否则返回false。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(N)
	3. 题解代码
```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.restcar = [[0] * 2 for _ in range(3)]
        self.restcar[0] = [big, 0]
        self.restcar[1] = [medium, 0]
        self.restcar[2] = [small, 0]

    def addCar(self, carType: int) -> bool:
        # print(self.restcar)
        if carType == 1:
            self.restcar[0][1] += 1
            if self.restcar[0][0] >= self.restcar[0][1]:
                return True
            else:
                return False
        elif carType == 2:
            self.restcar[1][1] += 1
            if self.restcar[1][0] >= self.restcar[1][1]:
                return True
            else:
                return False
        elif carType == 3:
            self.restcar[2][1] += 1
            if self.restcar[2][0] >= self.restcar[2][1]:
                return True
            else:
                return False
```

2. 一维数组
	1. 算法思路
		1. 创建一个一维数组park，初始park=[0,big,medium,small]。因为下标从0开始，故为后续方便使用，将大中小放在了下标的1，2，3位置。
		2. 利用cartype来定位park中对应车型剩余的位置个数：park[cartype]。当park[cartype]不为0时，返回true，同时-1，否则返回false。
	2. 复杂度分析
		1. 时间复杂度：O(1)
		2. 空间复杂度：O(1)
	3. 题解代码
```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.park = [0, big, medium, small]

    def addCar(self, carType: int) -> bool:
        if self.park[carType] == 0:
            return False
        else:
            self.park[carType] -= 1
            return True
```

#### 130.被围绕的区域（本周复习）
1. 深度优先搜索（DFS）
	1. 算法思路
		1. 根据题目，得知矩阵内有三种元素
			1. 字母X
			2. 被字母X包围的字母O
			3. 没有被字母X包围的字母O
		2. 根据解释，任何边界上的字母O都不会被包围。那么所有不被包围的O都是直接或间接与边界上的O相连。利用这个性质：从边界上的O开始将所有直接与间接相连的O都做好标记。
		3. 最后遍历矩阵，对于每个字母：
			1. 如果标记过，则说明该字母为没有被X包围的O，将其还原成O。
			2. 如果没有标记过，则说明该字母为被X包围的O，将其变为X。
		4. 第2点中，标记字母O的递归工作，与题目200.岛屿数量的递归方法类似。
	2. 复杂度分析
		1. 时间复杂度：O(m\*n)，m、n分别矩阵的行和列。
		2. 空间复杂度：O(m\*n)，递归工作时生成的系统栈。
	3. 题解代码
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None
        # board的长宽
        m, n = len(board), len(board[0])

        # dfs
        def dfs(board, x, y):
            # 终止条件
            if not 0 <= x < m or not 0 <= y < n or board[x][y] != 'O':
                return
            board[x][y] = 'A'
            dfs(board, x + 1, y)
            dfs(board, x - 1, y)
            dfs(board, x, y + 1)
            dfs(board, x, y - 1)

        # 找边界上的O
        for i in range(m):
            dfs(board, i, 0)
            dfs(board, i, n - 1)
        for j in range(n):
            dfs(board, 0, j)
            dfs(board, m - 1, j)
        # 更新board
        for x in range(m):
            for y in range(n):
                if board[x][y] == 'A':
                    board[x][y] = 'O'
                else:
                    board[x][y] = 'X'
```

2. 广度优先搜索（BFS）
	1. 算法思路与深度优先搜索（DFS）类似，不同之处在于将递归工作改成了维护一个队列，将与矩阵边界上的O直接或间接相连的O都放进了队列中进行标记。队列工作主要分为两步：
		1. 先遍历矩阵边界，将字母O的坐标记录在队列中。
		2. 弹出队列中字母O的坐标，将其进行标记，再遍历该坐标的相邻且在矩阵范围内的字母O的坐标，将其压入队列。不断迭代，直到将直接或间接相邻的字母O的坐标遍历完毕为止。
	2. 复杂度分析与深度优先搜索一致
	3. 题解代码
```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None
        # board的长宽
        m, n = len(board), len(board[0])

        # bfs
        queue = []
        for i in range(m):
            if board[i][0] == 'O':
                queue.append((i, 0))
            if board[i][n - 1] == 'O':
                queue.append((i, n - 1))
        for j in range(n):
            if board[0][j] == 'O':
                queue.append((0, j))
            if board[m - 1][j] == 'O':
                queue.append((m - 1, j))
        # 找相邻O
        while queue:
            x, y = queue.pop(0)
            board[x][y] = 'A'
            for mx, my in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                    queue.append((mx, my))

        # 更新board
        for x in range(m):
            for y in range(n):
                if board[x][y] == 'A':
                    board[x][y] = 'O'
                else:
                    board[x][y] = 'X'
```

3. 并查集
	1. 算法思路
		1. 与深度优先搜索类似，不同在于使用并查集的方法对矩阵内的所有节点做了连通分类。
		2. 并查集实现
			1. 虚拟一个dummy节点，作为没有被X包围的O的公共祖先
			2. 将矩阵内所有字母O添加进并查集内，边界字母O与dummy合并拥有相同的父节点；非边界字母O则有两种情况：
				1. 与边界O相连的，最终也与dummy拥有相同的父节点。
				2. 不与边界O相连的O，会互相连通，将其中一个O的父节点作为公共祖先，但与dummy的不一致。
		3. 最终遍历整个矩阵，判断每个字母的是否与dummy拥有相同的父节点，有则保留O，否则改为X。
	2. 复杂度分析
		1. 时间复杂度：O(m\*n)，遍历整个矩阵，并查集的时间复杂度约为O(1)。
		2. 空间复杂度：O(m)，并查集保存空间
	3. 题解代码
```python
# unionfind
class UnionFind:
    def __init__(self):
        self.father = {}

    def add(self, x):
        if x not in self.father:
            self.father[x] = None

    def find(self, x):
        root = x
        self.add(root)
        while self.father[root] is not None:
            root = self.father[root]
        # 路径压缩
        while x != root:
            o_father = self.father[x]
            self.father[x] = root
            x = o_father
        return root

    def merge(self, x, y):
        root_x, root_y = self.find(x), self.find(y)
        if root_x != root_y:
            self.father[root_x] = root_y

    def is_connected(self, x, y):
        return self.find(x) == self.find(y)


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        if not board:
            return None
        m, n = len(board), len(board[0])

        # 定义节点，用于将xy坐标转为一个节点
        def node(x, y):
            return x * n + y

        # dummy节点
        uf = UnionFind()
        dummy = m * n
        uf.add(dummy)
        # 遍历矩阵中的O
        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O':
                    # 边界O
                    if i == 0 or j == 0 or i == m - 1 or j == n - 1:
                        uf.merge(node(i, j), dummy)
                    else:  # 非边界O
                        for mx, my in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                            if 0 <= mx < m and 0 <= my < n and board[mx][my] == 'O':
                                uf.merge(node(i, j), node(mx, my))
        # 遍历矩阵的O
        # 判断与dummy的连通性
        for i in range(m):
            for j in range(n):
                if uf.is_connected(dummy, node(i, j)):
                    board[i][j] = 'O'
                else:
                    board[i][j] = 'X'
```

