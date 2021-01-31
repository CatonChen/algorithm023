# WEEK05学习笔记
本周其中考试，基础知识总结暂无。

### 本周leetcode练习总结
#### 127.单词接龙（本周复习）
1. 单向BFS
	1. 定义**st**：将**wordList**转换为一个**set集合**，达到**O(1)**的复杂度查询时间。
	2. 特例：如果**endWord**不在**st**中，则返回0，因为**wordList**中没有目标。
	3. 使用双端队列**deque**建立一个**queue**，将**beginWord**和计数器**step=1**加入队列。
	4. 建立一个**vistied**集合，用于记录变换的单词是否被访问过，初始将**beginWord**添加进集合。之后进行以下遍历递推：
		1. 当**queue**不为空时，取出首端元素**cur**和目标**endWord**比较是否相等，如果相等就返回**step**
		2. 当上述1不成立时，循环逐个替换**cur**中的字母生成**tmp**，如果**tmp**在集合**st**中，且未被访问过，则将**tmp**压入队列**queue**和集合**visited**中。
	5. 当所有可能都被遍历过后，如果没有返回**step**，则返回0，表示**wordList**中没有可变换至**endWord**的连接单词。
	6. 复杂度分析
		1. 时间复杂度：O(N)，N=m\*n，m是单词长度，n是wordList的长度
		2. 空间复杂度：O(N)，需要额外空间保存访问过的单词和队列
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        #初始化
        st=set(wordList)
        #特例
        if endWord not in st:
            return 0
        #初始化
        queue=collections.deque()
        queue.append((beginWord,1))
        visited=set()
        visited.add(beginWord)

        #遍历可能性
        while queue:
            cur, step = queue.popleft()
            if cur==endWord:
                return step
            #单词长度
            for i in range(len(cur)):
                #26个字母
                for j in range(26):
                    tmp = cur[:i]+chr(97+j)+cur[i+1:]
                    if tmp not in visited and tmp in st:
                        queue.append((tmp,step+1))
                        visited.add(tmp)

        return 0
```

2. 双向BFS
	1. 与上述单向BFS不同的在于，从**beginWord**和**endWord**同时开始做BFS搜索，将满足条件的单词分别加入**lqueue，lvisited，rqueue，rvisited**中。
	2. 以层为单位递增**step**
	3. 每次对元素少的队列进行搜索，如果访问的单词在另一侧已经访问过，说明接龙成功，返回**step**即可
	4. 复杂度分析
		1. 时间复杂度：O(N)，N=m\*n，m单词长度，n为wordList长度
		2. 空间复杂度：O(N)，需要额外空间保存访问过的单词和队列
```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # 初始化
        st = set(wordList)
        # 特例
        if endWord not in st:
            return 0

        from collections import deque
        # 初始化左右queue和vistied
        lqueue = deque()
        lqueue.append(beginWord)
        rqueue = deque()
        rqueue.append(endWord)

        lvisited = set()
        lvisited.add(beginWord)
        rvisited = set()
        rvisited.add(endWord)

        step = 0

        while lqueue and rqueue:
            # 找出哪个队列较短，交换
            if len(lqueue) > len(rqueue):
                lqueue, lvisited, rqueue, rvisited = rqueue, rvisited, lqueue, lvisited
            step += 1
            # 遍历较短的队列
            for k in range(len(lqueue)):
                cur = lqueue.popleft()
                if cur in rvisited:
                    return step
                else:
                    for i in range(len(cur)):
                        for j in range(26):
                            tmp = cur[:i] + chr(97 + j) + cur[i + 1:]
                            if tmp not in lvisited and tmp in st:
                                lqueue.append(tmp)
                                lvisited.add(tmp)

        return 0
```

#### 86.分隔链表（本周复习）
1. 双指针
	1. 创建两个空链表**p和q**
	2. 将小于**x**的链表部分记为**p**；大于**x**的链表部分记为**q**
	3. 最后拼接**p—\>q**
	4. 复杂度分析
		1. 时间复杂度：O(N)，n为链表长度
		2. 空间复杂度：O(N)，两个小链表保存原始链表的各自部分
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        p = less = ListNode(0)
        q = more = ListNode(0)

        while head:
            if head.val < x:
                less.next = head
                less = less.next
            else:
                more.next = head
                more = more.next
            head = head.next

        more.next = None
        less.next = q.next
        return p.next
```

#### 1.两数之和（本周复习）
1. 字典（哈希）
	1. 创建一个哈希表字典**hashmap**
	2. 遍历数组，在字典中寻找符合**target-nums[i]**的**key**，如果有符合的，则返回字典的**val（val是表示元素的下标）**和当前元素的**下标i**；否则将当前元素的值和下标加入**hashmap**中，其中**nums[i]**为字典的**key**，下标**i**为字典的**val**。
	3. 复杂度分析
		1. 时间复杂度：O(n)，最差情况遍历整个数组n
		2. 空间复杂度：O(n)，需要额外空间保存已遍历过的元素
```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = dict()
        for i, v in enumerate(nums):
            if hashmap.get(target - v) is not None:
                return [hashmap.get(target - v), i]
            hashmap[v] = i
```

#### 88.合并两个有序数组（本周复习）
1. 切片+排序
	1. 因为数组**nums1**拥有足够空间保存**nums2**，且元素长度为**m**，故通过对nums1做切片**nums1[:m]**，再并上nums2，即可得到一个完整的nums1，最后对新的nums1进行**排序**即可。
	2. 复杂度分析
		1. 时间复杂度：O(logN)，数组的排序耗时
		2. 空间复杂度：O(1)
```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        nums1[:]=nums1[:m]+nums2
        nums1.sort()
```

2. 三指针
	1. 因为num1和nums2是已经**排序**的数组，考虑设置**i , j , k**三个指针，并初始化为**i=m-1 , j=n-1 , k=m+n-1**。如此，就是从两个数组的最大值开始比较，并将**较大者**放置在**nums[k]**处。
	2. 当**nums[i]\>nums[j]**时，将nums[k]=nums[i]，同时**k-1 , i-1**；否则反之。
	3. 如果**j**先递减为**0**，那么返回num1即可，**因为nums1中剩余的元素肯定小于nums2**；如果**i**先递减为0，那么需要把nums2里剩余元素遍历后，依次加入nums1中。
	4. 复杂度分析
		1. 时间复杂度：O(N)，需要遍历两个数组，N=m+n
		2. 空间复杂度：O(1)，常量变量
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
        # 此处解决m=0 或 n=0 或 i=0&j<>0的情况
        if j >= 0:
            while j >= 0:
                nums1[k] = nums2[j]
                k -= 1
                j -= 1
```

#### 1128.等价多米诺骨牌对的数量
1. 字典（哈希表hashmap）
	1. 设计一种**list-\>int**的方式，使**domino[i][j]**中的较小者作10位数，较大者作个位数，比如**(1,2)或(2,1)都会有映射为数字12**，如此等价的domino骨牌都有相同的映射结果。
	2. 遍历一次数组，通过字典保存映射结果**\{映射结果：出现次数}**。
	3. 通过遍历字典，计数对数：**k\*(k-1)/2**
	4. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次数组
		2. 空间复杂度：O(N)，n为字典的缓存大小
```python
class Solution:
    def numEquivDominoPairs(self, dominoes: List[List[int]]) -> int:
        res = 0
        from collections import defaultdict
        dt = defaultdict(int)
        for i, j in dominoes:
            num = i * 10 + j if i < j else j * 10 + i
            dt[num] += 1

        for k in dt.values():
            res += int(k * (k - 1) / 2)
        return res
```

#### 874.模拟行走机器人（本周复习）
1. 贪心算法
	1. 创建一个字典：以**方向**作为key，value为一组数组，表示当前方向的**[x移动，y移动，当前方向的左侧，当前方向的右侧]**
	2. 将障碍物的坐标先**元组化**，再进行**set**处理
	3. 初始化变量**dir=‘up’（初始朝北），坐标x=y=0**
	4. 进行指令模拟
		1. 当**command=-2（左转） or -1（右转）**时，更新dir
		2. 当**command\>0**时模拟行走，**command**代表步数
			1. 如果行走路径的坐标匹配障碍物的坐标，则停止
			2. 否则返回欧式距离平方**x^2+y^2**
	5. 复杂度分析
		1. 时间复杂度：O(N)，n=command指令+障碍物坐标集合的长度
		2. 空间复杂度：O(N)，存储障碍物坐标集合的空间
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 初始化方向字典，方向为key， [x,y,当前方向左侧，当前方向右侧]为val
        direction = {
            'up': [0, 1, 'left', 'right'],
            'down': [0, -1, 'right', 'left'],
            'right': [1, 0, 'up', 'down'],
            'left': [-1, 0, 'down', 'up']
        }

        # 元组化障碍坐标
        obstacleset = set(map(tuple, obstacles))

        # 初始化坐标，方向和结果
        x = y = res = 0
        dir = 'up'

        # 模拟行走
        for cmd in commands:
            # 左转
            if cmd == -2:
                dir = direction[dir][2]  # 当前方向左侧
            # 右转
            elif cmd == -1:
                dir = direction[dir][3]  # 当前方向右侧
            elif cmd > 0:
                for i in range(cmd):
                    # 遇到障碍物
                    if (x + direction[dir][0], y + direction[dir][1]) in obstacleset:
                        break
                    else:   #正常行走
                        x += direction[dir][0]
                        y += direction[dir][1]
                        res = max(res, x ** 2 + y ** 2)
        return res
```

2. 贪心算法（另一种解法）
	1. 与第一种贪心算法不同的点：
		1. 因为机器人初始朝北，所以初始化坐标**x,y=0,0**，初始化步数**dx,dy=0,1**
		2. 当机器人左转时，步数变换**dx,dy=-dy,dx**；当右转时，步数变换**dx,dy=dy,-dx**
	2. 其他处理细节与第一种贪心算法类似
	3. 复杂度分析（与第一种贪心算法一致）
```python
class Solution:
    def robotSim(self, commands: List[int], obstacles: List[List[int]]) -> int:
        # 元组化障碍坐标
        obstacleset = set(map(tuple, obstacles))

        # 初始化坐标，方向和结果
        x = y = res = 0
        # 步数
        dx, dy = 0, 1

        # 模拟行走
        for cmd in commands:
            # 左转
            if cmd == -2:
                dx, dy = -dy, dx
            # 右转
            elif cmd == -1:
                dx, dy = dy, -dx
            elif cmd > 0:
                for i in range(cmd):
                    # 遇到障碍物
                    if (x + dx, y + dy) in obstacleset:
                        break
                    else:  # 正常行走
                        x += dx
                        y += dy
                        res = max(res, x ** 2 + y ** 2)
        return res
```

#### 94.二叉树的中序遍历（本周复习）
1. 递归法
	1. 根据中序遍历的顺序：**左-\>根-\>右**，调用函数自身进行递归
	2. 考虑特例：当根为空的时候，返回空即可
	3. 复杂度分析
		1. 时间复杂度：O(N)，n为二叉树的节点数，需要遍历所有节点
		2. 空间复杂度：O(logN)，平均情况为O(logN)，最差为O(N)
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        #递归
		#中序
        return self.inorderTraversal(root.left)+[root.val]+self.inorderTraversal(root.right)

		#前序
		# return [root.val]+self.inorderTraversal(root.left)+self.inorderTraversal(root.right)

		#后序
		# return self.inorderTraversal(root.left)+self.inorderTraversal(root.right)+[root.val]
```

2. 迭代（栈）
	1. 需要一个栈**stack**
	2. 先用指针找出每个节点（从root起）的最左下角，然后进行出栈
	3. 复杂度分析
		1. 时间复杂度：O(N)，n为二叉树的节点个数
		2. 空间复杂度：O(N)，n为树的高度
```python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        # 初始化
        res = []
        stack = []
        cur = root
        #中序遍历 
		#遍历栈或节点
        while stack or cur:
            # 当节点存在
            while cur:
                stack.append(cur)
                cur = cur.left  # 找节点的左节点，直到没有左节点为止
            cur = stack.pop()  # 出栈
            res.append(cur.val)
            cur = cur.right  # 找右节点
        return res

		# # 前序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.left
        #     cur = stack.pop()
        #     cur = cur.right
        # return res
        
        # # 后序，相同模板
        # while stack or cur:
        #     while cur:
        #         res.append(cur.val)
        #         stack.append(cur)
        #         cur = cur.right
        #     cur = stack.pop()
        #     cur = cur.left
        # return res[::-1]
```

#### 53.最大子序和
1. 贪心算法
	1. 设置最大值变量**max\_cur**和当前值变量**cur**，初始都为**nums[0]**
	2. 遍历数组，**cur取（num[i]+cur）与cur中较大者，后max\_cur取max\_cur与cur之间较大者**。
	3. 最终返回**max\_cur**。
	4. 复杂度分析
		1. 时间复杂度：O(N)，n为数组长度
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 贪心算法
        max_cur = cur = nums[0]  # 初始

        for i in range(1, len(nums)):
            # 取当前元素与当前元素之前和的较大者
            # 再取当前元素与之前最大值中的较大者
            cur = max(nums[i], cur + nums[i])
            max_cur = max(max_cur, cur)

        return max_cur
```

2. 动态规划
	1. 遍历数组，**如果nums[i-1]\>0，则nums[i]=nums[i]+nums[i-1]**，否则保持nums[i]不变；最终返回数组nums中的最大值。
	2. 复杂度分析
		1. 时间复杂度：O(N)，遍历数组的长度
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 动态规划
        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
        # print(nums)
        return max(nums)
```

#### 98.验证二叉搜索树（本周复习）
1. 利用中序遍历的特性
	1. 根据二叉搜索树的特性：**左节点\<根节点\<右节点，且每个节点的左右子树同样具有该特点**。
	2. 二叉搜索树中序遍历出来的数组是一个**升序数组**的特点。
	3. 利用栈的**后入先出**特点，将节点的左节点逐个压入栈，直到最深的左子节点为止；最终**最后入栈的元素应为所有左节点中最小者**。
	4. 不断的弹出栈顶元素，让其与前一个栈顶元素做比较，**如果当前栈顶元素\<=前一个栈顶元素**，则表示不是一个有效的二叉搜索树，返回false。当栈里所有元素都弹出且不满足这个条件后，返回true。
	5. 复杂度分析
		1. 时间复杂度：O(N)，遍历二叉树的节点数
		2. 空间复杂度：O(N)，栈维护的空间，二叉树的深度
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        stack = []
        inorder = float('-inf')

        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            # print(stack)
            node = stack.pop()
            # print(node.val)
            if node.val <= inorder:
                return False
            inorder = node.val
            root = node.right
        return True
```

2. 递归+生序数组判断
	1. 先利用**递归**生成中序遍历的结果
	2. 利用二叉搜索树中序遍历的结果为一个**生序数组**的特点，判断是否为有效二叉搜索树
	3. 复杂度分析
		1. 时间复杂度：O(N)，遍历树的所有节点
		2. 空间复杂度：O(N)，1.递归时系统生成的栈空间，2.遍历结果数组
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        # 递归生成中序遍历结果
        def recur(node):
            if not node:
                return []
            return recur(node.left) + [node.val] + recur(node.right)

        res = recur(root)
        # print(res)
        return all(res[i] > res[i - 1] for i in range(1, len(res)))
```

#### 74.搜索二维矩阵
1. 一维数组转换
	1. 根据矩阵的特性（如下），将矩阵转换为一个**生序的一维数组**。
		1. 每一行都是生序整数
		2. 每一行的第一个整数大于前一行的最后一个整数
	2. 利用**二分查找**对一维数组搜索目标值，有返回true，否则false。
	3. 二分查找方法
		1. 设置**left=0**和**right=len(nums)-1**的左右边界，**mid=(left+right)/2**。
		2. 三种情况处理：
			1. 当**nums[mid]=target**时，表示找到目标，返回true
			2. 当**nums[mid]\<target**时，表示目标在mid的**右侧**，移动left
			3. 当**nums[mid]\>target**时，表示目标在mid的**左侧**，移动right
	4. 复杂度分析
		1. 时间复杂度：O(M\*N)，矩阵转换为一维数据需要m\*n，m为行，n为列；二分查找需要O(logN)。
		2. 空间复杂度：O(N)，保存一维数据的空间
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 特例
        if not matrix or not matrix[0]:
            return False

        # 转为一维数组
        nums = [i for row in matrix for i in row]
        # print(nums)

        # 二分查找
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
```

#### 1143.最长公共子序列
1. 动态规划
	1. 将text1和text2视作一个**矩阵**，假设**下标i**表示矩阵的横轴，**下标j**表示矩阵的纵轴
	2. 当**text1[i]==text2[j]**时，矩阵**dp[i][j]=dp[i][j]+1**
	3. 当**text1[i]\<\>text2[j]**时，则让**dp[i][j]取dp[i-1][j]和dp[i][j-1]中的较大者，进行状态转移**。
	4. 最终返回**dp[m][n]，m,n=text1,text2**的长度
	5. 复杂度分析
		1. 时间复杂度：O(M\*N)，text1和text2的长度乘积
		2. 空间复杂度：O(M\*N)，保存了m\*n的矩阵空间
```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        # 初始化长度
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for i in range(m + 1)]
        # print(dp)
        for i in range(1, m+1):
            for j in range(1, n+1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        # print(dp)
        return dp[m][n]
```

#### 102.二叉树的层序遍历（本周复习）
1. 广度优先搜索（BFS）
	1. 利用队列先进先出的特点，逐层将每层的节点压入队列，再依次弹出队列的首端元素。
	2. 如果弹出的元素有左右子节点，就将左右子节点压入队列，放入下次队列遍历。
	3. 需要一个变量保存当前队列的长度，表示本层遍历完毕。
	4. 复杂度分析
		1. 时间复杂度：O(N)，n为树的所有节点
		2. 空间复杂度：O(N)，队列保存了树的所有节点
```python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        from collections import deque
        res = []
        # 初始化队列
        queue = deque()
        queue.append(root)

        while queue:
            n = len(queue)  # 本层队列长度
            tmp = []
            for i in range(n):  # 遍历本层队列的元素
                node = queue.popleft()
                if not node:  # node无效，则继续
                    continue
                tmp.append(node.val)
                # 压入node的子节点
                queue.append(node.left)
                queue.append(node.right)
            # print(tmp)
            if tmp:
                res.append(tmp)

        return res
```

2. 深度优先搜索（DFS）
	1. 创建一个递归函数dfs用于递归节点的左子树和右子树，注意以下几点：
		1. 终止条件：当传入节点为空时，返回
		2. **当最终结果res的长度等于当前层次level时**，即遍历到一个新层次，需要创建新列表保存结果。
		3. 将递归的节点保存在res对应的**level**的列表里。
		4. 复杂度分析
			1. 时间复杂度：O(N)，递归树的所有节点
			2. 空间复杂度：O(N)，递归时系统调用的栈
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    # 递归
    def dfs(self, root, level, res):
        # 终止条件
        if root is None:
            return
        # 当到新的一层时，要创建一个新数组保存结果
        if len(res) == level:
            res.append([])
        res[level].append(root.val)  # 把当层结果加入当层的数组里
        # 递归工作
        self.dfs(root.left, level + 1, res)
        self.dfs(root.right, level + 1, res)

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []
        self.dfs(root, 0, res)
        return res
```

#### 18.四数之和
1. 双指针遍历
	1. 与三数之和类似，但有几点需注意：
		1. 在三数之和外再嵌套一层循环：假设**p是外循环，k是内循环**，p和k表示的是四数里的前两个数。
		2. 因为目标值target是任意数，所以与三数之和不同的是有以下几个条件要注意：
			1. [nums\[p]+nums\[p+1]+nums\[p+2]+nums\[p+3]\>target]()时**break**
			2. [nums\[p]+nums\[n-1]+nums\[n-2]+nums\[n-3]\<target]()时，做**continue**，取下一个p的操作
			3. 保持[p\>0 and nums\[p]==nums\[p-1]]()时，做**continue**，取下一个p的操作
			4. 同理，对k也是如此处理。
		3. p和k处理后，令**i=k+1，j=n-1**，当**i\<j**时对剩余的两个数求解，此处与三数之和不同的点有：
			1. 当[nums\[p]+nums\[k]+nums\[i]+nums\[j]==target]()时，当i\<j成立时，
				1. [while i\<j and nums\[i]==nums\[i+1]]()时，i+=1，非**nums[i]==nums[i-1]**
				2. 同理，[while i\<j and nums\[j]==nums\[j-1]]()时，j-=1，非**nums[j]==nums[j+1]**
		4. 复杂度分析
			1. 时间复杂度：O(N\*\*3)，n为数组长度，枚举四元组需要n\*n\*n次，排序需要O(NlogN)
			2. 空间复杂度：O(N)
```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums)
        res = []
        nums.sort()
        if not nums or n < 4:
            return res

        for p in range(n - 3):
            if p > 0 and nums[p] == nums[p - 1]:
                continue
            if nums[p] + nums[p + 1] + nums[p + 2] + nums[p + 3] > target:
                break
            if nums[p] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target:
                continue
            # k = p + 1
            for k in range(p + 1, n - 2):
                if k > p + 1 and nums[k] == nums[k - 1]:
                    continue
                if nums[p] + nums[k] + nums[k + 1] + nums[k + 2] > target:
                    break
                if nums[p] + nums[k] + nums[n - 1] + nums[n - 2] < target:
                    continue
                i, j = k + 1, n - 1
                while i < j:
                    s = nums[p] + nums[k] + nums[i] + nums[j]
                    if s == target:
                        res.append([nums[p], nums[k], nums[i], nums[j]])
                        while i < j and nums[i] == nums[i + 1]:  # 条件与三数之和不同
                            i += 1
                        i += 1
                        while i < j and nums[j] == nums[j - 1]:  # 条件与三数之和不同
                            j -= 1
                        j -= 1
                    elif s > target:
                        j -= 1
                    else:
                        i += 1
        return res
```

2. 递归法
	1. 将问题拆解至每次都求两数之和是否等于目标值
	2. 对于两数之和，设定左右指针i,j，通过i右移，j左移来寻找num[i]+nums[j]=target，找到就将结果添加到res中
	3. 对于三数之和，依次从数组中选择一个数，并固定它，然后转换为求剩余两个数的两数之和问题
	4. 对于四数之和或n数之和，参考1、2、3点，通过递归工作的方式将其转换为求两数之和的问题
	5. 注意跳过重复数据
	6. 复杂度分析
		1. 时间复杂度：O(N^n)，两数之和是O(N\*2)，实际看需要分解成几个两数之和的子问题
		2. 空间复杂度：O(N)，递归时调用n个系统栈？
```python
class Solution:
    def nSum(self, nums, n, target):
        if len(nums) < n:
            return []

        res = []
        # 分解到n为2时，求2数之和等于目标
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
        else:  # 否则继续分解
            for i in range(len(nums)):
                if i > 0 and nums[i] == nums[i - 1]:
                    continue
                subres = self.nSum(nums[i + 1:], n - 1, target - nums[i])
                for j in range(len(subres)):
                    res.append([nums[i]] + subres[j])
            return res

    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        if len(nums) < 4:
            return []
        nums.sort()
        return self.nSum(nums, 4, target)
```

#### 74.搜索二维矩阵（本周复习）
1. 矩阵一位数组化
	1. 先将二维矩阵转换为一维数组，再通过二分查找的方法搜索目标
		1. 转换一位数组：**nums = [i for row in matrix for i in row]**
		2. 寻找目标直接套用二分查找代码模板
	2. 复杂度分析
		1. 时间复杂度：O(M\*N)，m和n为矩阵行和列，二分查找需要O(logN)的时间
		2. 空间复杂度：O(N)，n为一位数组的空间
```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 特例
        if not matrix or not matrix[0]:
            return False

        # 转为一维数组
        nums = [i for row in matrix for i in row]

        # 二分查找
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return False
```

#### 45.跳跃游戏II
1. 贪心算法
	1. 参数定义
		1. **cur** :当前跳的最大范围
		2. **nex** ：下一跳的最远距离
		3. **step** ：最小步数
	2. 算法思路
		1. 每一跳在**cur**范围内活动，并更新可以跳到的最远距离**nex**
		2. 如果**下标i超过cur**，令cur=nex，开始下一跳，**step+1**
		3. 当**nex\>n-1**时，返回**step+1**
	3. 复杂度分析
		1. 时间复杂度：O(N)，n为数组长度
		2. 空间复杂度：O(1)，常数变量
```python
lass Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        #特例
        if n == 1:
            return 0
        cur = nex = step = 0  # cur当前跳的最大范围，nex下一跳最远距离
        for i, v in enumerate(nums):
            # print(i,v,cur)
            if i > cur:
                cur = nex
                # print(i,cur)
                step += 1
            nex = max(nex, i + v)
            # print(nex)
            if nex >= n - 1:
                return step + 1
```

#### 104.二叉树的最大深度（本周复习）
1. 递归法
	1. 当root为空时，返回0
	2. 递归工作，调用自身函数，将节点的左节点传入：**left=self.maxDepth(root.left)** ；
	3. 同理可得 **right=self.maxDepth(root.right)**
	4. 返回**max(left,right)+1**即可
	5. 复杂度分析
		1. 时间复杂度：O(N)，n为二叉树的节点个数
		2. 空间复杂度：O(N)，n为树的深度
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: TreeNode) -> int:
        if not root:
            return 0
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return max(left, right) + 1
```

#### 105.从前序与中序遍历序列构造二叉树（本周复习）
1. 递归法
	1. 根据前序（根-\>左-\>右）和中序（左-\>根-\>右）的遍历方式，找出根节点root在中序序列中的位置inorder\_root
	2. 根据根节点root在中序遍历中的位置inorder\_root，得到左子树的个数len\_left，由此可得
		1. 左子树在preorder中为[1,1+len\_left]，在inorder中为[0,inorder\_root-1]
		2. 右子树在preorder中为[0,1+len\_left+1,len(preorder)]，在inorder中为[inorder\_root+1,len(inorder)]
		3. 最后返回root
	3. 由递归工作反复进行2点，直到整个二叉树构造完成。
	4. 复杂度分析
		1. 时间复杂度：O(N)，n为树的所有节点个数
		2. 空间复杂度：O(N)，递归时的系统栈
```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        def myBuildTree(pre_left, pre_right, in_left, in_right):
            # 终止条件
            if pre_left > pre_right:
                return None
            # 前序根下标
            pre_root = preorder[pre_left]
            # 中序根下标
            in_root = index[pre_root]

            root = TreeNode(pre_root)

            len_left = in_root - in_left

            root.left = myBuildTree(pre_left + 1, pre_left + len_left, in_left, in_root - 1)
            root.right = myBuildTree(pre_left + len_left + 1, pre_right, in_root + 1, in_right)
            return root

        n = len(preorder)
        index = {ele: i for i, ele in enumerate(inorder)}
        return myBuildTree(0, n - 1, 0, n - 1)
```

#### 剑指 offer 05.替换空格
1. 迭代
	1. 枚举字符串s，**i代表当前字符**，res为新空字符串。
	2. **如果i=‘ ’时，则替换成‘%20’，并加入res**；否则就将i加入res。
	3. 复杂度分析
		1. 时间复杂度：O(N)，遍历一次字符串
		2. 空间复杂度：O(1)，常数变量
```python
class Solution:
    def replaceSpace(self, s: str) -> str:
        res = ''
        for i in s:
            if i == ' ':
                res += '%20'
            else:
                res += i
        return res
```

#### 62.不同路径
1. 动态规划
	1. 从左上角移动到**(i,j)**的路径数量，其中i,j的范围是**[0,m-1)和[0,n-1)**。
	2. 每次只能向下移动一步或向右移动一步，如果要到达(i,j)，那么：
		1. 如果向下走一步，则从**(i-1,j)**过来
		2. 如果向右走一步，则从**(i,j-1)**过来
	3. 得到状态转移公式：**f(i,j)=f(i-1,j)+f(i,j-1)**
	4. 需注意当i=j=0时，该公式不满足，所以初始条件**f(0,0)=1**
	5. 最终得到公式**f(m-1,n-1)**
	6. **细节：将f(0,j)和f(i,0)作为边界，其值都设为1**
	7. 复杂度分析
		1. 时间复杂度：O(M\*N)，m,n为矩阵的长宽
		2. 空间复杂度：O(M\*N)，利用了m\*n的空间
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # m=i , n=j
        # 矩阵f
        f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        # print(f)
        for i in range(1, m):
            for j in range(1, n):
                f[i][j] = f[i - 1][j] + f[i][j - 1]

        return f[m - 1][n - 1]
```

2. 动态规划（空间优化）
	1. 在上述1的解法上，进行空间压缩。
	2. 根据规律，得出公式：f[j]=f[j]+f[j-1]
	3. 复杂度分析
		1. 时间复杂度：O(M\*N)
		2. 空间复杂度：O(N)
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        # n=j
        f = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                f[j] += f[j - 1]
        return f[n - 1]
```


