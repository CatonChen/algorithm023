# 字典 wordList 中从单词 beginWord 和 endWord 的 转换序列 是一个按下述规格形成的序列： 
# 
#  
#  序列中第一个单词是 beginWord 。 
#  序列中最后一个单词是 endWord 。 
#  每次转换只能改变一个字母。 
#  转换过程中的中间单词必须是字典 wordList 中的单词。 
#  
# 
#  给你两个单词 beginWord 和 endWord 和一个字典 wordList ，找到从 beginWord 到 endWord 的最短转换序列中的单
# 词数目。如果不存在这样的转换序列，返回 0。 
#  
# 
#  示例 1： 
# 
#  
# 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","lo
# g","cog"]
# 输出：5
# 解释：一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog", 返回它的长度 5。
#  
# 
#  示例 2： 
# 
#  
# 输入：beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","lo
# g"]
# 输出：0
# 解释：endWord "cog" 不在字典中，所以无法进行转换。 
# 
#  
# 
#  提示： 
# 
#  
#  1 <= beginWord.length <= 10 
#  endWord.length == beginWord.length 
#  1 <= wordList.length <= 5000 
#  wordList[i].length == beginWord.length 
#  beginWord、endWord 和 wordList[i] 由小写英文字母组成 
#  beginWord != endWord 
#  wordList 中的所有字符串 互不相同 
#  
#  Related Topics 广度优先搜索 
#  👍 682 👎 0


# leetcode submit region begin(Prohibit modification and deletion)
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

# leetcode submit region end(Prohibit modification and deletion)
