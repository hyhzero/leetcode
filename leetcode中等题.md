## 中等题

### 1. 在排序数组中查找元素的第一个和最后一个位置

* 二分法

  ```python
  class Solution:
      def searchRange(self, nums: List[int], target: int) -> List[int]:
  
          left = 0
          right = len(nums)-1
          while left <= right:
              mid = (left + right)//2
  
              if nums[mid]==target:
                  break
              elif nums[mid] < target:
                  left = mid + 1
              else:
                  right = mid - 1
          
          if left > right:
              return [-1, -1]
  
          left = mid
          while left >= 0 and nums[left]==target:
              left -= 1
  
          right = mid
          while right < len(nums) and nums[right]==target:
              right += 1
  
          return [left+1, right-1]
  ```

* 改进的二分法

  ```python
  class Solution:
      def searchRange(self, nums: List[int], target: int) -> List[int]:
  
          left = 0
          right = len(nums)-1
          ans = []
          while left <= right:
              mid = (left + right)//2
  
              if nums[mid]==target:
                  right = mid - 1
              elif nums[mid] < target:
                  left = mid + 1
              else:
                  right = mid - 1
          
          if left < len(nums) and nums[left]==target:
              ans.append(left)
          else:
              return [-1, -1]
  
          left = 0
          right = len(nums)-1
          while left <= right:
              mid = (left + right)//2
  
              if nums[mid]==target:
                  left = mid + 1
              elif nums[mid] < target:
                  left = mid + 1
              else:
                  right = mid - 1
  
          if nums[right]==target:
              ans.append(right)
  
          return ans
  
  ```

### 2. 下一个排列

* 全排列非递归

  ```python
  class Solution:
      def nextPermutation(self, nums: List[int]) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
  
          i = len(nums)-1
          while i > 0 and nums[i] <= nums[i-1]:
              i -= 1
  
          if i <= 0:
              left = 0
              right = len(nums)-1
              while left < right:
                  temp = nums[left] 
                  nums[left] = nums[right]
                  nums[right] = temp
  
                  left += 1
                  right -= 1
          else:
  
              idx = i
              j = i
              while j < len(nums):
                  if nums[j] > nums[i-1] and nums[idx] >= nums[j]:
                      idx = j
  
                  j += 1
              temp = nums[i-1]
              nums[i-1] = nums[idx]
              nums[idx] = temp
  
              left = i
              right = len(nums)-1
              while left < right:
                  temp = nums[left]
                  nums[left] = nums[right]
                  nums[right] = temp
  
                  left += 1
                  right -= 1
  ```

### 3. 电话号码的字母组合

* 深度优先搜索

  ```python
  class Solution:
  
      def dfs(self, digits, i, nums, opts, ans):
          if i==len(digits):
              ans.append(''.join(nums[:]))
              return
  
          idx = ord(digits[i])-ord('2')
          for op in opts[idx]:
              nums[i] = op
              self.dfs(digits, i+1, nums, opts, ans)
  
  
      def letterCombinations(self, digits: str) -> List[str]:
          if len(digits)==0:
              return []
  
          opts = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
          ans = []
          nums = list(digits)
          self.dfs(digits, 0, nums, opts, ans)
  
          return ans
  ```

* 回溯非递归

  ```python
  class Solution:
  
      def letterCombinations(self, digits: str) -> List[str]:
          if len(digits)==0:
              return []
      
          opts = ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
          nums = [0 for _ in range(len(digits))]
          cap = [len(opts[ord(d)-ord('2')]) for d in digits]
          ans = []
          row = 0
          while True:
              
              if nums[row]==cap[row] and row==0:
                  break
  
              if nums[row]==cap[row]:
                  nums[row] = 0
                  row -= 1
                  nums[row] += 1
              else:
                  row += 1
  
              if row==len(nums):
                  ans.append(''.join([opts[ord(digits[i])-ord('2')][nums[i]] for i in range(len(digits))]))
                  row -= 1
                  nums[row] += 1
  
          return ans
  ```

### 4. 删除链表的倒数第N个节点

* 线性扫描

  ```python
  
  class Solution:
      def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
  
          if head==None:
              return None
          
          p = head
          q = head
          for _ in range(n):
              p = p.next
  
          if p==None:
              return head.next
  
          while p and p.next:
              p = p.next
              q = q.next
          
          q.next = q.next.next
          return head
  ```

### 5. 最接近的三个数之和

* 排序和双指针

  ```python
  class Solution:
      def threeSumClosest(self, nums: List[int], target: int) -> int:
          if len(nums) < 3:
              return None
  
          nums.sort()
          dist = abs(sum(nums[:3])-target)
          ans = sum(nums[:3])
          for i in range(len(nums)-2):
  
              left = i + 1
              right = len(nums)-1
  
              while left < right:
                  sums = nums[i] + nums[left] + nums[right]
  
                  if abs(sums-target) < dist:
                      dist = abs(sums-target)
                      ans = sums
                  if sums > target:
                      right -= 1
                  else:
                      left += 1
  
          return ans
  ```


### 6. 整数转罗马数字

* 字符映射

  ```python
  class Solution:
      def intToRoman(self, num: int) -> str:
  
          nums = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
          roman = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC', 'C', 'CD', 'D', 'CM','M']
          hashmap = dict(zip(roman, nums))
  
          ans = []
          i = len(nums)-1
          while num > 0:
              
              while i >= 0 and num < nums[i]:
                  i -= 1
  
              n = num//nums[i]
              num %= nums[i]
              ans.append(n*roman[i])
  
          return ''.join(ans)
  ```

  

### 7. 三数之和

* 排序和双指针

  ```python
  class Solution:
      def threeSum(self, nums: List[int]) -> List[List[int]]:
  
          if len(nums) < 3:
              return []
  
          ans = []
          nums.sort()
  
          
          for i in range(len(nums)-2):
              if i > 0 and nums[i]==nums[i-1]:
                  continue
  
              left = i + 1
              right = len(nums)-1
              while left < right:
                  if nums[i]+nums[left]+nums[right] == 0:
                      ans.append([nums[i], nums[left], nums[right]])
                      left += 1
                      while left < right and nums[left]==nums[left-1]:
                          left += 1
                  elif nums[i]+nums[left]+nums[right] < 0:
                      left += 1
                  else:
                      right -= 1
  
          return ans
  ```

### 8. 四数之和

* 排序和双指针

  ```python
  class Solution:
      def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
          if len(nums) < 4:
              return []
  
          ans = []
          nums.sort()
          for i in range(len(nums)-3):
              if i > 0 and nums[i]==nums[i-1]:
                  continue
  
              for j in range(i+1, len(nums)-2):
                  if j > (i+1) and nums[j]==nums[j-1]:
                      continue
  
                  left = j + 1
                  right = len(nums)-1
                  while left < right:
                      if (nums[i] + nums[j] + nums[left] + nums[right])==target:
                          ans.append([nums[i], nums[j], nums[left], nums[right]])
                          left += 1
                          while left < right and nums[left]==nums[left-1]:
                              left += 1
                      elif (nums[i] + nums[j] + nums[left] + nums[right])<target:
                          left += 1
                      else:
                          right -= 1
  
          return ans
  ```

### 9. 盛最多水的容器

* 双指针法

  ```python
  class Solution:
      def maxArea(self, height: List[int]) -> int:
          
          left = 0
          right = len(height)-1
          ans = 0
          while left < right:
              ans = max(ans, min(height[left], height[right])*(right-left))
              if height[left] < height[right]:
                  left += 1
              else:
                  right -= 1
              
  
          return ans
  ```

### 10. 括号生成

* 回溯法（递归）

  ```python
  class Solution:
  
      def valid(self, nums, i):
          if sum(nums[:i]) < 0 or nums.count(1)*2 > len(nums):
              return False
  
          return True
  
      def dfs(self, nums, i, ans):
          if i==len(nums):
              ans.append(''.join(['(' if x==1 else ')' for x in nums[:]]))
              return
          
          for v in [1, -1]:
              nums[i] = v
              if self.valid(nums, i):
                  self.dfs(nums, i+1, ans)
  
  
      def generateParenthesis(self, n: int) -> List[str]:
  
          ans = []
          nums = [0 for _ in range(2*n)]
          self.dfs(nums, 0, ans)
          return ans
  ```

* 动态规划

  ```python
  class Solution:
  
      def generateParenthesis(self, n: int) -> List[str]:
  
          dp = [[] for _ in range(n+1)]
          dp[0] = ['']
          dp[1] = ['()']
          for i in range(2, n+1):
              for j in range(i):
  
                  for k in range(len(dp[j])):
                      for s in range(len(dp[i-1-j])):
                          dp[i].append('({}){}'.format(dp[j][k], dp[i-1-j][s]))
  
          return dp[-1]
  ```


### 11. 两两交换链表中的节点

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def swapPairs(self, head: ListNode) -> ListNode:
          if head==None or head.next==None:
              return head
  
          node = ListNode(0)
          node.next = head
          p = node
          q = head.next
  
          while q:
              t1 = p.next
              t2 = q.next
              p.next = q
              q.next = t1
              t1.next = t2
              q = t1
  
              if q.next and q.next.next:
                  q = q.next.next
                  p = t1
              else:
                  break
  
          return node.next
  ```

* 递归

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
  
      def swap(self, head):
          if head==None or head.next==None:
              return head
          
          node1 = head
          node2 = head.next
  
          node1.next = self.swap(node2.next)
          node2.next = node1
          
          return node2
  
      def swapPairs(self, head: ListNode) -> ListNode:
          
          return self.swap(head)
  ```

### 12. 全排列II

* 交换法

  ```python
  class Solution:
  
  
      def permuteUnique(self, nums: List[int]) -> List[List[int]]:
  
          nums.sort()
          ans = []
          while True:
              ans.append(nums[:])
              
              i = len(nums)-1
              while i > 0 and nums[i] <= nums[i-1]:
                  i -= 1
              
              if i==0:
                  break
  
              idx = i
              j = i
              while j < len(nums):
                  if nums[j] > nums[i-1] and nums[j] <= nums[idx]:
                      idx = j
  
                  j += 1
              
              temp = nums[idx]
              nums[idx] = nums[i-1]
              nums[i-1] = temp
  
              left = i
              right = len(nums)-1
              while left < right:
                  temp = nums[left] 
                  nums[left] = nums[right]
                  nums[right] = temp
  
                  left += 1
                  right -= 1
  ```

* 回溯法

  ```python
  class Solution:
  
      def valid(self, nums, i, cond):
          for j in range(i):
              if cond[i]==cond[j] or nums[i]==nums[j] and cond[i] < cond[j]:
                  return False
          return True
  
      def perm(self, nums, i, opts, cond, ans):
          if i==len(nums):
              ans.append(nums[:])
              return
  
          for j in range(len(opts)):
              nums[i] = opts[j]
              cond[i] = j
              if self.valid(nums, i, cond):
                  self.perm(nums, i+1, opts, cond, ans)
  
      def permuteUnique(self, nums: List[int]) -> List[List[int]]:
  
          ans = []
          opts = tuple(nums)
          nums = [0 for _ in range(len(nums))]
          cond = [0 for _ in range(len(nums))]
          self.perm(nums, 0, opts, cond, ans)
          return ans
  ```

### 13. 有效的数独

* 暴力

  ```python
  class Solution:
      def isValidSudoku(self, board: List[List[str]]) -> bool:
  
          
          for i in range(len(board)):
              counts = [0 for _ in range(10)]
              for j in range(len(board[0])):
                  if board[i][j].isdigit():
                      counts[ord(board[i][j])-ord('0')] += 1
                      if counts[ord(board[i][j])-ord('0')] > 1:
                          return False
  
  
          for j in range(len(board[0])):
              counts = [0 for _ in range(10)]
              for i in range(len(board)):
                  if board[i][j].isdigit():
                      counts[ord(board[i][j])-ord('0')] += 1
                      if counts[ord(board[i][j])-ord('0')] > 1:
                          return False
  
          for j in range(len(board[0])):
              counts = [0 for _ in range(10)]
              for i in range(len(board)):
                  if board[i][j].isdigit():
                      counts[ord(board[i][j])-ord('0')] += 1
                      if counts[ord(board[i][j])-ord('0')] > 1:
                          return False
  
          for x in range(0, len(board), 3):
              for y in range(0, len(board[0]), 3):
                  counts = [0 for _ in range(10)]
                  for i in range(x, x + 3):
                      for j in range(y, y + 3):
                          if board[i][j].isdigit():
                              counts[ord(board[i][j])-ord('0')] += 1
                              if counts[ord(board[i][j])-ord('0')] > 1:
                                  return False
  
          return True
          
  ```


### 14. 组合总和

* 回溯法

  ```python
  class Solution:
  
      def valid(self, nums, target):
          if target < 0:
              return False
  
          if max(nums)!=nums[-1]:
              return False
          
          return True
  
      def dfs(self, nums, opts, target, ans):
          if target==0:
              ans.append(nums[:])
              return
  
          for op in opts:
              nums.append(op)
              if self.valid(nums, target):
                  self.dfs(nums, opts, target-op, ans)
              nums.pop()
  
  
      def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
          opts = tuple(candidates)
          nums = []
          ans = []
          self.dfs(nums, opts, target, ans)
          return ans
  ```

  

### 15. 第K个排列

* 交换法

  ```python
  class Solution:
      def getPermutation(self, n: int, k: int) -> str:
  
          nums = [i for i in range(1, n+1)]
  
          for _ in range(k-1):
  
              i = len(nums)-1
              while i > 0 and nums[i] <= nums[i-1]:
                  i -= 1
              
              idx = i
              j = i
              for j in range(i, len(nums)):
                  if nums[j] > nums[i-1] and nums[j] <= nums[idx]:
                      idx = j
              
              temp = nums[idx]
              nums[idx] = nums[i-1]
              nums[i-1] = temp
  
              left = i
              right = len(nums)-1
              while left < right:
                  temp = nums[left]
                  nums[left] = nums[right]
                  nums[right] = temp
  
                  left += 1
                  right -= 1
  
          return ''.join(list(map(str, nums)))
  ```

### 16. 丑数II

* 动态规划

  ```python
  class Solution:
      def nthUglyNumber(self, n: int) -> int:
  
          dp = [1 for _ in range(n)]
          i2, i3, i5 = 0, 0, 0
  
          for i in range(1, n):
              minv = min(dp[i2]*2, dp[i3]*3, dp[i5]*5)
              dp[i] = minv
              if dp[i2]*2==minv:
                  i2 += 1
              if dp[i3]*3==minv:
                  i3 += 1
              if dp[i5]*5==minv:
                  i5 += 1
  
          return dp[-1]
  ```

  

### 17. 用 Rand7() 实现 Rand10()

* 二次采样

  ```python
  # The rand7() API is already defined for you.
  # def rand7():
  # @return a random integer in the range 1 to 7
  
  class Solution:
      def rand10(self):
          """
          :rtype: int
          """
  
          while True:
  
              row = rand7()
              col = rand7()
              v = col + (row-1)*7
  
              if v <= 40:
                  return 1 + v%10
  ```

### 18. 预测赢家

* 动态规划

  ```python
  class Solution:
      def PredictTheWinner(self, nums: List[int]) -> bool:
  
  
          row = len(nums)
          col = len(nums)
          opt = [[0 for _ in range(col)] for _ in range(row)]
  
          for i in range(row):
              for j in range(col):
                  if i==j:
                      opt[i][j] = nums[i]
                  if j==i+1:
                      opt[i][j] = max(nums[i], nums[j])
  
          for j in range(2, col):
              x, y = 0, j
              while x < row and y < col:
  
                  opt1 = nums[x] + min(opt[x+2][y], opt[x+1][y-1])
                  opt2 = nums[y] + min(opt[x][y-2], opt[x+1][y-1])
                  opt[x][y] = max(opt1, opt2)
                  x += 1
                  y += 1
          
          return opt[0][col-1]*2 >= sum(nums)
  ```

* 动态规划2

  ```pythoh
  class Solution:
      def PredictTheWinner(self, nums: List[int]) -> bool:
  
  
          row = len(nums)
          col = len(nums)
          opt = [[0 for _ in range(col)] for _ in range(row)]
  
          for i in range(row):
              for j in range(col):
                  if i==j:
                      opt[i][j] = nums[i]
  
          for j in range(1, col):
              x, y = 0, j
              while x < row and y < col:
                  opt[x][y] = max(nums[x]-opt[x+1][y], nums[y]-opt[x][y-1])
                  
                  x += 1
                  y += 1
          
          return opt[0][-1] >= 0
  ```


### 19. 验证IP地址

* 暴力

  ```python
  class Solution:
      def validIPAddress(self, IP: str) -> str:
  
          ipv4 = IP.split('.')
          ipv6 = IP.split(':')
          if len(ipv4)!=4 and len(ipv6)!=8:
              return "Neither"
          
          if len(ipv4)==4 and len(ipv6)==1:
              for ip in ipv4:
                  if not ip.isdigit() or ip[0]=='0' and len(ip) > 1 or int(ip) > 255:
                      return 'Neither'
              
              return 'IPv4'
          elif len(ipv4)==1 and len(ipv6)==8:
              for ip in ipv6:
                  if len(ip) > 4 or len(ip)==0:
                      return "Neither"
                  
                  for i in range(len(ip)):
                      if ip[i] not in list('0123456789abcdefABCDEF'):
                          return "Neither"
              return "IPv6"
          else:
              return "Neither"
  ```

### 20. 目标和

* 回溯法（超时）

  ```python
  class Solution:
  
      def findTargetSumWays(self, nums: List[int], S: int) -> int:
  
          vals = [0  for _ in range(len(nums))]
  
          self.ans = 0
  
          def dfs(vals, nums, i, S):
              if i==len(vals):
                  if sum(vals)==S:
                      self.ans += 1
                  return
  
              for sign in [1, -1]:
                  vals[i] = nums[i]*sign
                  dfs(vals, nums, i+1, S)
  
          dfs(vals, nums, 0, S)
          
          return self.ans
  ```

* 分治法(超时)

  ```python
  class Solution:
  
      def sumway(self, nums, i, S):
          if i==len(nums) and S==0:
              return 1
          if i==len(nums) and S!=0:
              return 0
          
          c1 = self.sumway(nums, i+1, S-nums[i])
          c2 = self.sumway(nums, i+1, S+nums[i])
          return c1 + c2
  
      def findTargetSumWays(self, nums: List[int], S: int) -> int:
  
          return self.sumway(nums, 0, S)
  ```



### 21. 提莫攻击

* 暴力

  ```python
  class Solution:
      def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
          if len(timeSeries)==0:
              return 0
          if len(timeSeries)==1:
              return duration
  
          i = 0
          sums = 0
          while i < len(timeSeries)-1:
              if timeSeries[i+1]-timeSeries[i] < duration:
                  sums += timeSeries[i+1]-timeSeries[i]
              else:
                  sums += duration
  
              i += 1
  
          sums += duration
          return sums
  ```

  

### 22. 零钱兑换

* 回溯法(超时)

  ```python
  class Solution:
  
  
      def dfs(self, nums, coins, amount):
          if amount < 0:
              return
          if amount==0:
              self.ans += 1
              return
          
          for c in coins:
              if len(nums) > 0 and nums[-1] > c:
                  continue
              
              nums.append(c)
              self.dfs(nums, coins, amount-c)
              nums.pop()
  
      def change(self, amount: int, coins: List[int]) -> int:
  
          self.ans = 0
          self.dfs([], coins, amount)
          return self.ans
  ```

* 动态规划

  ```python
  class Solution:
  
      def change(self, amount: int, coins: List[int]) -> int:
  
          opts = [0 for _ in range(amount+1)]
          opts[0] = 1
          
          for coin in coins:
              for v in range(coin, amount+1):
                  opts[v] += opts[v-coin]
  
          return opts[-1]
  ```

### 23. 对角线遍历

* 模拟扫描

  ```python
  class Solution:
      def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:
  
          direct = 0
          x, y = 0, 0
          ans = []
          while x < len(matrix) and y < len(matrix[0]):
              
              if direct==0:
                  while x >= 0 and y < len(matrix[0]):
                      ans.append(matrix[x][y])
                      x -= 1
                      y += 1
                  x += 1
                  y -= 1
                  if y==len(matrix[0])-1:
                      x += 1
                  else:
                      y += 1
              else:
                  while y >= 0 and x < len(matrix):
                      ans.append(matrix[x][y])
                      x += 1
                      y -= 1
                  x -= 1
                  y += 1
                  if x==len(matrix)-1:
                      y += 1
                  else:
                      x += 1
  
              if x==len(matrix)-1 and y==len(matrix[0])-1:
                  ans.append(matrix[x][y])
                  break
              
              direct = (direct + 1)%2
  
          return ans
  ```

* 对角线迭代和翻转

  ```python
  class Solution:
      def findDiagonalOrder(self, matrix: List[List[int]]) -> List[int]:
  
          ans = []
          x, y = 0, 0
          direct = 0
          while x < len(matrix) and y < len(matrix[0]):
  
              i, j = x, y
              temp = []
              while i < len(matrix) and j >= 0:
                  temp.append(matrix[i][j])
                  i += 1
                  j -= 1
              
              if direct==0:
                  ans.extend(reversed(temp))
              else:
                  ans.extend(temp)
              
              direct = (direct + 1)%2
  
              if x==len(matrix)-1 and y==len(matrix[0])-1:
                  break
              
              if y==len(matrix[0])-1:
                  x += 1
              else:
                  y += 1
  
          return ans
  ```

### 24. 汉明距离总和

* 计数法

  ```python
  class Solution:
      def totalHammingDistance(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
  
          maxv = max(nums)
          count = 0
          while maxv > 0:
              maxv //= 2
              count += 1
          
          c0 = [0 for _ in range(count)]
          c1 = [0 for _ in range(count)]
  
          
          ans = 0
          for num in nums:
              for i in range(count):
                  if num%2==0:
                      c0[i] += 1
                  else:
                      c1[i] += 1
                  num //= 2
              
          for i in range(count):
              ans += c0[i]*c1[i]
          
          return ans
  ```

### 25. 下一个更大的元素

* 单调栈

  ```python
  class Solution:
      def nextGreaterElements(self, nums: List[int]) -> List[int]:
          if len(nums)==0:
              return []
          if len(nums)==1:
              return [-1]
  
  
          stack = []
          stack.append((nums[0], 0))
          ans = [None for _ in range(len(nums))]
  
          for i in range(1, 2*len(nums)-1):
              i %= len(nums)
              while len(stack) > 0 and stack[-1][0] < nums[i]:
                  _, idx = stack.pop()
                  if ans[idx]==None:
                      ans[idx] = nums[i]
              
              stack.append((nums[i], i))
  
          while len(stack) > 0:
              val, i = stack.pop()
              if ans[i]==None:
                  ans[i] = -1
  
          return ans
  ```

### 26. 最长回文子序列

* 动态规划

  ```python
  class Solution:
      def longestPalindromeSubseq(self, s: str) -> int:
  
          dp = [[0 for _ in range(len(s))] for _ in range(len(s))]
  
          for i in range(len(s)):
              dp[i][i] = 1
          for i in range(len(s)):
              if i < len(s)-1:
                  if s[i]==s[i+1]:
                      dp[i][i+1] = 2
                  else:
                      dp[i][i+1] = 1
          
          for j in range(2, len(s)):
              
              i = 0
              while j < len(s):
                  if s[i]==s[j]:
                      dp[i][j] = 2 + dp[i+1][j-1]
                  else:
                      dp[i][j] = max(dp[i][j-1], dp[i+1][j])
                  i += 1
                  j += 1
          return dp[0][len(s)-1]
  ```


### 27. 跳跃游戏

* 动态规划

  ```python
  class Solution:
  
      def canJump(self, nums: List[int]) -> bool:
          if len(nums)==1:
              return True
          if len(nums)==0 or nums[0]==0:
              return False
  
          opts = [0 for _ in range(len(nums))]
          opts[0] = 1
          
          for i in range(1, len(nums)):
              j = i-1
              while j >= 0:
                  if opts[j]==1 and (j + nums[j]) >= i:
                      opts[i] = 1
                      break
          
                  j -= 1
          
          return opts[-1]==1
  ```

* 贪心

  ```python
  class Solution:
  
      
      def canJump(self, nums: List[int]) -> bool:
          if len(nums)==0:
              return False
  
          f = 0
          for i in range(len(nums)):
              if i > f:
                  return False
              
              f = max(f, i+nums[i])
  
          return True
  ```

  

### 28. 旋转矩阵

* 模拟

  ```python
  class Solution:
      def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          if len(matrix)==0:
              return []
  
          ans = []
          visit = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
          r, c = 0, 0
          R, C = len(matrix), len(matrix[0])
  
          dr = [0, 1, 0, -1]
          dc = [1, 0, -1, 0]
          di = 0
  
          for _ in range(R*C):
              ans.append(matrix[r][c])
              visit[r][c] = True
              rc = r + dr[di]
              cc = c + dc[di]
  
              if rc < 0 or rc >= R or cc < 0 or cc >= C or visit[rc][cc]==True:
                  di = (di + 1)%4
                  r += dr[di]
                  c += dc[di]
              else:
                  r = rc
                  c = cc
  
          return ans
  ```

* 按层遍历

  ```python
  class Solution:
      def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
          if len(matrix)==0:
              return []
  
          r1, c1, r2, c2 = 0, 0, len(matrix)-1, len(matrix[0])-1
          ans = []
          while r1 <= r2 and c1 <= c2:
  
              for j in range(c1, c2+1):
                  ans.append(matrix[r1][j])
              for i in range(r1+1, r2+1):
                  ans.append(matrix[i][c2])
              
              if r1 < r2 and c1 < c2:
                  for j in range(c2-1, c1, -1):
                      ans.append(matrix[r2][j])
                  for i in range(r2, r1, -1):
                      ans.append(matrix[i][c1])
  
              r1 += 1
              c1 += 1
              r2 -= 1
              c2 -= 1
  
          return ans
  ```

  

### 29. 颜色分类

* 两边扫描

  ```python
  class Solution:
      def sortColors(self, nums: List[int]) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
  
          i = 0
          for j in range(len(nums)):
              if nums[j]==0:
                  temp = nums[j]
                  nums[j] = nums[i]
                  nums[i] = temp
                  i += 1
  
          for j in range(len(nums)):
              if nums[j]==1:
                  temp = nums[j]
                  nums[j] = nums[i]
                  nums[i] = temp
                  i += 1
  ```

* 颜色分类

  ```python
  class Solution:
      def sortColors(self, nums: List[int]) -> None:
          """
          Do not return anything, modify nums in-place instead.
          """
  
          left = 0
          right = len(nums)-1
          i = 0
          while i >= left and i <= right:
              if nums[i]==0:
                  temp = nums[i]
                  nums[i] = nums[left]
                  nums[left] = temp
                  left += 1
                  i += 1
              elif nums[i]==2:
                  temp = nums[right]
                  nums[right] = nums[i]
                  nums[i] = temp
                  right -= 1
              else:
                  i += 1
  ```

### 30. 组合

* 回溯法

  ```python
  class Solution:
  
      def valid(self, nums, i):
          if i==0:
              return True
          if nums[i] <= nums[i-1]:
              return False
          
          return True
          
  
      def dfs(self, nums, i, ans, n):
          if i==len(nums):
              ans.append(nums[:])
              return
  
          for v in range(1, n+1):
              nums[i] = v
              if self.valid(nums, i):
                  self.dfs(nums, i+1, ans, n)
                  
  
      def combine(self, n: int, k: int) -> List[List[int]]:
  
          ans = []
          nums = [0 for _ in range(k)]
          self.dfs(nums, 0, ans, n)
          return ans
  ```

### 31. 合并区间

* 排序

  ```python
  class Solution:
      def merge(self, intervals: List[List[int]]) -> List[List[int]]:
          if len(intervals)==0:
              return []
  
          intervals = list(sorted(intervals, key=lambda x: x[0]))
  
          ans = []
          i = 0
          prev = intervals[0]
          while i < len(intervals):
  
              while i < len(intervals) and  intervals[i][0] <= prev[1]:
                  prev[1] = max(intervals[i][1], prev[1])
                  i += 1
              
              if i < len(intervals):
                  ans.append(prev)
                  prev = intervals[i]
              else:
                  ans.append(prev)
  
          return ans
  ```

### 32. 旋转链表

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def rotateRight(self, head: ListNode, k: int) -> ListNode:
          if head==None or head.next==None:
              return head
          
          n = 0
          p = head
          while p and p.next:
              n += 1
              p = p.next
          n += 1
          tail = p
          
          k %= n
          if k==0:
              return head
              
          p = head
          for _ in range(n-k-1):
              p = p.next
  
          newhead = p.next
          tail.next = head
          p.next = None
          return newhead
  ```


### 33. 旋转图像

* 暴力

  ```python
  class Solution:
      def rotate(self, matrix: List[List[int]]) -> None:
          """
          Do not return anything, modify matrix in-place instead.
          """
  
          for i in range(len(matrix)//2):
              for j in range(i, len(matrix[0])-1-i):
                  r, c = i, j
                  prev = matrix[r][c]
                  for _ in range(4):
                      r, c = c, len(matrix)-1-r
                      temp = matrix[r][c]
                      matrix[r][c] = prev
                      prev = temp
  ```

* 转置+反转

  ```python
  class Solution:
      def rotate(self, matrix: List[List[int]]) -> None:
          """
          Do not return anything, modify matrix in-place instead.
          """
  
          # 转置
          for i in range(len(matrix)):
              for j in range(i, len(matrix[0])):
                  temp = matrix[i][j]
                  matrix[i][j] = matrix[j][i]
                  matrix[j][i] = temp
          
          # 反转
          for i in range(len(matrix)):
              for j in range(len(matrix[0])//2):
                  temp = matrix[i][j]
                  matrix[i][j] = matrix[i][len(matrix)-1-j]
                  matrix[i][len(matrix)-1-j] = temp
  ```

### 34. 删除排序数组中的重复元素II

* 暴力

  ```python
  class Solution:
      def removeDuplicates(self, nums: List[int]) -> int:
  
          i = 0
          k = 0
          while i < len(nums):
  
              j = i
              while j < len(nums) and nums[j]==nums[i]:
                  j += 1
  
              if j-i >= 2:
                  nums[k] = nums[i]
                  nums[k+1] = nums[i]
                  k += 2
              elif j-i==1:
                  nums[k] = nums[i]
                  k += 1
  
              i = j
  
          return k
  ```

  

### 35. 单词接龙（未完成）

* 回溯法（超时）

  ```python
  class Solution:
  
  
      def valid(self, words, i):
  
          for j in range(i):
              if words[j]==words[i]:
                  return False
          
          count = 0
          for k in range(len(words[i])):
              if words[i-1][k]!=words[i][k]:
                  count += 1
          if count > 1:
              return False
          else:
              return True
  
  
      def dfs(self, words, i, wordList, target, ans):
          if i > 0 and words[i-1]==target:
              ans.append(i)
              return
  
  
          for w in wordList:
              words[i] = w
              if self.valid(words, i):
                  self.dfs(words, i+1, wordList, target, ans)
  
  
      def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
          if endWord not in wordList or beginWord==endWord:
              return 0
          
  
          words = [beginWord for _ in range(len(wordList)+1)]
          ans = []
          self.dfs(words, 1, wordList, endWord, ans)
          return min(ans) if len(ans) > 0 else 0
  ```

  

### 36. Pow(x, n)（未完）

* 贪心

  ```python
  class Solution:
      def myPow(self, x: float, n: int) -> float:
          if n==0:
              return 1
          if n==1:
              return x
  
          sign = 1 if n > 0 else -1
          n = abs(n)
          pows = [1]
          vals = [x]
          while pows[-1] < n:
              
              i = len(pows)-1
              while i >= 0 and pows[i]+pows[-1] > n:
                  i -= 1
  
              vals.append(vals[i]*vals[-1])
              pows.append(pows[i] + pows[-1])
  
              if pows[-1]==n:
                  break
  
          return vals[-1] if sign > 0 else 1/vals[-1]
  
  ```

* 快速幂算法

  ```python
  class Solution:
  
      def pows(self, x, n):
          if n==0:
              return 1
          if n==1:
              return x
          
          half = self.pows(x, n//2)
          if n%2==0:
              return half*half
          else:
              return x*half*half
  
      def myPow(self, x: float, n: int) -> float:
          if n >= 0:
              return self.pows(x, n)
          else:
              return 1/self.pows(x, abs(n))
  ```


### 37. 分割回文串(未完)

* 回溯法

  ```python
  class Solution:
  
      def hw(self, s):
          i = 0
          j = len(s)-1
          while i < j and s[i]==s[j]:
              i += 1
              j -= 1
          
          return i >= j
  
      def dfs(self, strs, s, ans):
          if s=='':
              ans.append(strs[:])
              return
  
          i = 1
          while i <= len(s):
              if self.hw(s[:i]):
                  strs.append(s[:i])
                  self.dfs(strs, s[i:], ans)
                  strs.pop()
              i += 1
  
  
      def partition(self, s: str) -> List[List[str]]:
  
          ans = []
          strs = []
          self.dfs(strs, s, ans)
          return ans
  ```

### 38. 最长回文子串

* 动态规划

  ```python
  class Solution:
  
      def longestPalindrome(self, s: str) -> str:
          if len(s) <= 1:
              return s
          
          dp = [[True for _ in range(len(s))] for _ in range(len(s))]
  
          d = 0
          ans = s[0]
          for j in range(1, len(s)):
              i = 0
              while j < len(s):
                  dp[i][j] = (s[i]==s[j]) and dp[i+1][j-1]
                  if dp[i][j]==True and j-i > d:
                      ans = s[i:j+1]
                      d = j-i
                  
                  i += 1
                  j += 1
  
          return ans
  ```

### 39. 子集(未完)

* 回溯法

  ```python
  class Solution:
  
  
      def dfs(self, opts, i, nums, ans):
          if i==len(opts):
              ans.append([nums[j] for j in range(len(opts)) if opts[j]==1])
              return
  
          for v in [0, 1]:
              opts[i] = v
              self.dfs(opts, i+1, nums, ans)
  
      def subsets(self, nums: List[int]) -> List[List[int]]:
  
          ans = []
          opts = [0 for _ in range(len(nums))]
          self.dfs(opts, 0, nums, ans)
          return ans
  ```

### 40. 乘积最大子序列

* 动态规划

  ```python
  class Solution:
      def maxProduct(self, nums: List[int]) -> int:
  
          if len(nums)==1:
              return nums[0]
          
          imax = [0 for _ in range(len(nums))]
          imin = [0 for _ in range(len(nums))]
  
          imax[0] = nums[0]
          imin[0] = nums[0]
          ans = nums[0]
          for i in range(1, len(nums)):
              imax[i] = max([nums[i], nums[i]*imax[i-1], nums[i]*imin[i-1]])
              imin[i] = min([nums[i], nums[i]*imax[i-1], nums[i]*imin[i-1]])
  
              ans = max(ans, imax[i])
  
          return ans
  ```


### 41. 组合总数II(未完)

* 回溯法

  ```python
  class Solution:
  
      
      def dfs(self, bucket, i, opts, target, ans):
          if target==0:
              ans.append([opts[j] for j in range(i) if bucket[j]==1])
              return
  
          if target < 0 or i >= len(opts):
              return
          
          bucket[i] = 0
          self.dfs(bucket, i+1, opts, target, ans)
          
          for j in range(i):
              if bucket[j]==0 and opts[j]==opts[i]:
                  return
  
          bucket[i] = 1
          self.dfs(bucket, i+1, opts, target-opts[i], ans)
          
  
      def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
          candidates.sort()
          opts = tuple(candidates)
          ans = []
          bucket = [0 for _ in range(len(opts))]
          self.dfs(bucket, 0, opts, target, ans)
          return ans
  ```

* 回溯法2

  ```python
  class Solution:
  
      def valid(self, bucket, i, opts, target):
          if bucket[i]==0:
              return True
  
          for j in range(i):
              if bucket[j]==0 and opts[j]==opts[i]:
                  return False
  
          if target - opts[i] < 0:
              return False
  
          return True
          
      
      def dfs(self, bucket, i, opts, target, ans):
          if target==0:
              ans.append([opts[j] for j in range(i) if bucket[j]==1])
              return
  
          if i==len(bucket):
              return
          
          for v in [0, 1]:
              bucket[i] = v
              if self.valid(bucket, i, opts, target):
                  target = target if v==0 else target-opts[i]
                  self.dfs(bucket, i+1, opts, target, ans)
          
  
      def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
          candidates.sort()
          opts = tuple(candidates)
          ans = []
          bucket = [0 for _ in range(len(opts))]
          self.dfs(bucket, 0, opts, target, ans)
          return ans
  ```

  

### 42. 格雷编码

* 镜像法

  ```python
  class Solution:
      def grayCode(self, n: int) -> List[int]:
          if n==0:
              return [0]
          if n==1:
              return [0, 1]
  
          g = [0, 1]
          inc = 2
          for _ in range(1, n):
              ans = g[:]
              for val in g[::-1]:
                  ans.append(inc + val)
              g = ans[:]
              inc *= 2
  
          return ans
  ```

* 镜像法2

  ```python
  class Solution:
      def grayCode(self, n: int) -> List[int]:
          if n==0:
              return [0]
          if n==1:
              return [0, 1]
  
          ans = [0, 1]
          inc = 2
          for _ in range(1, n):
              for i in range(len(ans)-1, -1, -1):
                  ans.append(inc + ans[i])
              inc *= 2
  
          return ans
  ```

### 43. 子集II

* 回溯法

  ```python
  class Solution:
  
      def valid(self, opts, i, nums):
          if opts[i]==0:
              return True
  
          for j in range(len(opts)):
              if opts[j]==0 and nums[j]==nums[i]:
                  return False
  
          return True
  
  
      def dfs(self, opts, i, nums, ans):
          if i==len(nums):
              ans.append([nums[j] for j in range(len(opts)) if opts[j]==1])
              return
  
          for opt in [0, 1]:
              opts[i] = opt
              if self.valid(opts, i, nums):
                  self.dfs(opts, i+1, nums, ans)
  
      def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
  
          opts = [0 for _ in range(len(nums))]
          ans = []
          self.dfs(opts, 0, nums, ans)
          return ans
  ```

### 44. 组合总数

* 回溯法

  ```python
  class Solution:
  
      def valid(self, nums, target):
          if target < nums[-1]:
              return False
          
          if len(nums) > 1 and nums[-1] < nums[-2]:
              return False
          
          return True
  
      def dfs(self, nums, opts, target, ans):
          if target==0:
              ans.append(nums[:])
              return
          
          if target < 0:
              return
  
          for op in opts:
              nums.append(op)
              if self.valid(nums, target):
                  self.dfs(nums, opts, target-op, ans)
              nums.pop()
  
  
      def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
          opts = tuple(candidates)
          nums = []
          ans = []
          self.dfs(nums, opts, target, ans)
          return ans
  ```

  

### 45. 复原IP地址

* 回溯法

  ```python
  class Solution:
  
      def dfs(self, ip, i, s, ans):
          if i==len(s):
              if len(ip)==4:
                  ans.append('.'.join(ip))
              return
          
          if len(ip) > 4:
              return
  
          if s[i]=='0':
              ip.append('0')
              self.dfs(ip, i+1, s, ans)
              ip.pop()
              return
  
          for j in range(i, len(s)):
              val = int(s[i:j+1])
              if val > 255:
                  break
  
              ip.append(s[i:j+1])
              self.dfs(ip, j+1, s, ans)
              ip.pop()
              
              
              
              
  
      def restoreIpAddresses(self, s: str) -> List[str]:
          ip = []
          ans = []
          self.dfs(ip, 0, s, ans)
          return ans
  ```

* 回溯法2

  ```python
  class Solution:
  
      def dfs(self, bucket, i, j, s, ans):
          if i==len(bucket) and j==len(s):
              ans.append('.'.join(bucket))
              return
          
          if i==len(bucket) or j==len(s):
              return 
          
          k = j
          while k < len(s) and k < j+3:
              val = int(s[j:k+1])
              if val > 255:
                  break
              
              if s[j]=='0':
                  bucket[i] = '0'
                  self.dfs(bucket, i+1, j+1, s, ans)
                  break
              
              bucket[i] = s[j:k+1]
              self.dfs(bucket, i+1, k+1, s, ans)
              k += 1       
              
  
      def restoreIpAddresses(self, s: str) -> List[str]:
          bucket = [0 for _ in range(4)]
          ans = []
          self.dfs(bucket, 0, 0, s, ans)
          return ans
  ```

  

### 46. 组合总数III

* 回溯法

  ```python
  class Solution:
  
      def valid(self, nums, i, target):
          if target < nums[i]:
              return False
  
          if i > 0 and nums[i] <= nums[i-1]:
              return False
  
          return True
  
  
      def dfs(self, nums, i, target, ans):
          if i==len(nums):
              if target==0:
                  ans.append(nums[:])
              return
  
  
          for v in range(1, 10):
              nums[i] = v
              if self.valid(nums, i, target):
                  self.dfs(nums, i+1, target-v, ans)
  
  
      def combinationSum3(self, k: int, n: int) -> List[List[int]]:
  
          ans = []
          nums = [None for _ in range(k)]
          self.dfs(nums, 0, n, ans)
          return ans
  ```

  

### 47. 累加数

* 回溯法

  ```python
  class Solution:
  
      def dfs(self, f1, f2, i, num):
          if i==len(num) and f1!=None and f2!=None and str(f1)+str(f2)!=num:
              return True
          
  
          for j in range(i, len(num)):
              s = num[i:j+1]
              
              if s[0]=='0' and len(s) > 1:
                  continue
              
              if f1==None:
                  res = self.dfs(int(s), f2, j+1, num)
              elif f2==None:
                  res = self.dfs(f1, int(s), j+1, num)
              elif f1+f2==int(s):
                  res = self.dfs(f2, int(s), j+1, num)
              else:
                  res = False
              
              if res==True:
                  return True
  
          return False
          
  
  
  
  
      def isAdditiveNumber(self, num: str) -> bool:
          if len(num) < 3:
              return False
          
          return self.dfs(None, None, 0, num)
  ```

### 48. 计算各个位数不同的数字个数(未完)

* 回溯法

  ```python
  class Solution:
  
  
      def valid(self, nums, i):
          j = 0
          while j < i and nums[j]==0:
              j += 1
          
          if j >= i:
              return True
          
          while j < i and nums[j]!=nums[i]:
              j += 1
  
          return j >= i
  
  
      def dfs(self, nums, i):
          if i==len(nums):
              return 1
  
          count = 0
          for val in range(10):
              nums[i] = val
              if self.valid(nums, i):
                  count += self.dfs(nums, i+1)
  
          return count
                  
  
  
      def countNumbersWithUniqueDigits(self, n: int) -> int:
  
          nums = [None for _ in range(n)]
          return self.dfs(nums, 0)
  
  ```

### 49. 优美的排列

* 回溯法

  ```python
  class Solution:
  
  
      def valid(self, nums, i):
          if nums[i]%(i+1)!=0 and (i+1)%nums[i]!=0:
              return False
              
          for j in range(i):
              if nums[j]==nums[i]:
                  return False
          
          return True
  
  
      def dfs(self, nums, i, n):
          if i==len(nums):
              return 1
          
          count = 0
          for val in range(1, n+1):
              nums[i] = val
              if self.valid(nums, i):
                  count += self.dfs(nums, i+1, n)
  
          return count
  
              
  
  
      def countArrangement(self, N: int) -> int:
          nums = [None for _ in range(N)]
          return self.dfs(nums, 0, N)
  ```


### 50. 不同路径

* 动态规划

  ```python
  class Solution:
      def uniquePaths(self, m: int, n: int) -> int:
          if m==1 or n==1:
              return 1
  
          dp = [[0 for _ in range(n)] for _ in range(m)]
          
          for j in range(n):
              dp[0][j] = 1
          for i in range(m):
              dp[i][0] = 1
  
          for j in range(1, n):
              i = 0
              while i < m and j >= 0:
                  if dp[i][j]==0:
                      if i >= 1:
                          dp[i][j] += dp[i-1][j]
                      if j >= 1:
                          dp[i][j] += dp[i][j-1]
                  i += 1
                  j -= 1
  
          for i in range(1, m):
              j = n-1
              while i < m and j >= 0:
                  if dp[i][j]==0:
                      if i >= 1:
                          dp[i][j] += dp[i-1][j]
                      if j >= 1:
                          dp[i][j] += dp[i][j-1]
                  i += 1
                  j -= 1
          
          return dp[-1][-1]
  
  
  ```

* 动态规划优化

  ```python
  class Solution:
  
      
      def uniquePaths(self, m: int, n: int) -> int:
          dp = [1]*n
          for _ in range(m):
              for i in range(1, n):
                  dp[i] += dp[i-1]
  
          return dp[-1]
  ```

### 51. 不同路径II

* 动态规划

  ```python
  class Solution:
      def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
          if obstacleGrid[0][0]==1:
              return 0
  
          for i in range(len(obstacleGrid)):
              for j in range(len(obstacleGrid[0])):
                  if obstacleGrid[i][j]==1:
                      obstacleGrid[i][j] = -1
  
          obstacleGrid[0][0] = 1
          for i in range(len(obstacleGrid)):
              for j in range(len(obstacleGrid[0])):
                  if obstacleGrid[i][j]==0:
                      if i > 0 and obstacleGrid[i-1][j]!=-1:
                          obstacleGrid[i][j] += obstacleGrid[i-1][j]
                      if j > 0 and obstacleGrid[i][j-1]!=-1:
                          obstacleGrid[i][j] += obstacleGrid[i][j-1]
  
          return obstacleGrid[-1][-1] if obstacleGrid[-1][-1]!=-1 else 0
  ```

* 动态规划优化

  ```python
  class Solution:
      def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
          if obstacleGrid[0][0]==1:
              return 0
          
          dp = [0]*len(obstacleGrid[0])
          dp[0] = 1
          for i in range(len(obstacleGrid)):
              for j in range(len(obstacleGrid[0])):
                  if obstacleGrid[i][j]==0:
                      if j > 0:
                          dp[j] += dp[j-1]
                  else:
                      dp[j] = 0
  
          return dp[-1]
  ```

### 52. 最小路径和

* 动态规划

  ```python
  class Solution:
      def minPathSum(self, grid: List[List[int]]) -> int:
  
          dp = [0]*len(grid[0])
          sums = 0
          for j in range(len(grid[0])):
              sums += grid[0][j]
              dp[j] = sums
  
          for i in range(1, len(grid)):
              for j in range(len(grid[0])):
                  if j==0:
                      dp[j] += grid[i][j]
                  else:
                      dp[j] = grid[i][j] + min(dp[j], dp[j-1])
  
          return dp[-1]
  ```

### 53. 不同的二叉搜索树

* 动态规划

  ```python
  class Solution:
      def numTrees(self, n: int) -> int:
          if n==1:
              return 1
  
          opt = [0 for _ in range(n+1)]
          opt[0] = 1
          opt[1] = 1
          for i in range(2, n+1):
              for j in range(i):
                  opt[i] += opt[j]*opt[i-j-1]
  
          return opt[-1]
  ```

### 54. 三角形最小路径和

* 动态规划

  ```python
  class Solution:
      def minimumTotal(self, triangle: List[List[int]]) -> int:
  
          dp = [0]*len(triangle)
          dp[0] = triangle[0][0]
          for i in range(1, len(triangle)):
              for j in range(len(triangle[i])-1, -1, -1):
                  if j==len(triangle[i])-1:
                      dp[j] = dp[j-1] + triangle[i][j]
                  elif j==0:
                      dp[j] += triangle[i][j]
                  else:
                      dp[j] = min(dp[j], dp[j-1]) + triangle[i][j]
  
          return min(dp)
  ```

  

### 55. 打家劫舍II

* 动态规划

  ```python
  class Solution:
      def rob(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
  
          if len(nums)==1:
              return nums[0]
          if len(nums)==2:
              return max(nums)
  
          dp = [0 for _ in range(len(nums))]
          dp[0] = 0
          dp[1] = nums[1]
          for i in range(2, len(nums)):
              dp[i] = max(dp[i-1], dp[i-2]+nums[i])
          
          ans = dp[-1]
  
          dp[0] = nums[0]
          dp[1] = max(nums[0], nums[1])
          for i in range(2, len(nums)-1):
              dp[i] = max(dp[i-1], dp[i-2]+nums[i])
  
          ans = max(ans, dp[-2])
          return ans
  ```


### 56. 最大正方形

* 动态规划

  ```python
  class Solution:
      def maximalSquare(self, matrix: List[List[str]]) -> int:
          if len(matrix)==0:
              return 0
  
          dp = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
  
          ans = 0
          for i in range(len(matrix)):
              dp[i][0] = 0 if matrix[i][0]=='0' else 1
              ans = max(ans, dp[i][0])
          for j in range(len(matrix[0])):
              dp[0][j] = 0 if matrix[0][j]=='0' else 1
              ans = max(ans, dp[0][j])
  
          for i in range(1, len(matrix)):
              for j in range(1, len(matrix[0])):
                  if matrix[i][j]=='0':
                      dp[i][j] = 0
                  else:
                      dp[i][j] = min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
  
                  ans = max(ans, dp[i][j])
  
  
          return ans**2
  ```

* 动态规划，空间优化

  ```python
  class Solution:
      def maximalSquare(self, matrix: List[List[str]]) -> int:
          if len(matrix)==0:
              return 0
  
          ans = 0
          dp = [0 for _ in range(len(matrix[0]))]
          for i in range(len(matrix[0])):
              dp[i] = 0 if matrix[0][i]=='0' else 1
              ans = max(ans, dp[i])
  
          for i in range(1, len(matrix)):
              temp = 0 if matrix[i][0]=='0' else 1
              for j in range(1, len(matrix[0])):
                  if matrix[i][j]=='1':
                      target = min(temp, dp[j-1], dp[j]) + 1
                  else:
                      target = 0
                  dp[j-1] = temp
                  ans = max(ans, temp)
                  temp = target
              dp[-1] = temp
              ans = max(ans, temp)
  
          return ans**2
  ```

### 57. 解码方法

* 动态规划

  ```python
  class Solution:
      def numDecodings(self, s: str) -> int:
          if s[0]=='0':
              return 0
  
          dp = [0 for _ in range(len(s)+1)]
  
          dp[0] = 1
          dp[1] = 1
          for i in range(2, len(s)+1):
              x = ord(s[i-2])-ord('0')
              y = ord(s[i-1])-ord('0')
              if y==0:
                  if x==0 or x >= 3:
                      return 0
                  dp[i] = dp[i-2]
              else:
                  if x==0:
                      dp[i] = dp[i-1]
                  else:
                      temp = x*10 + y
                      if temp <= 26:
                          dp[i] = dp[i-1] + dp[i-2]
                      else:
                          dp[i] = dp[i-1]
  
          return dp[-1]
  ```

### 58. 完全平方数

* 动态规划(超时)

  ```python
  class Solution:
      def numSquares(self, n: int) -> int:
          if n==1:
              return 1
  
          dp = [0 for _ in range(n+1)]
          dp[0] = 0
          dp[1] = 1
          for i in range(2, n+1):
              temp = dp[i-1] + 1
              for j in range(1, i+1):
                  if j*j <= i:
                      temp = min(temp, dp[i-j*j]+1)
              dp[i] = temp
          
          return dp[-1]
  ```



### 59. 最长上升子序列

* 动态规划

  ```python
  class Solution:
      def lengthOfLIS(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
  
          ans = 1
          dp = [0 for _ in range(len(nums))]
          dp[0] = 1
          for i in range(1, len(nums)):
              max_v = 0
              for j in range(i):
                  if nums[j] < nums[i]:
                      max_v = max(dp[j], max_v)
              
              dp[i] = max_v + 1
  
              ans = max(ans, dp[i])
  
          return ans
  ```


### 60. 零钱兑换

* 动态规划

  ```python
  class Solution:
      def coinChange(self, coins: List[int], amount: int) -> int:
  
          if amount==0:
              return 0
  
          dp = [0 for _ in range(amount+1)]
          for coin in coins:
              if coin < len(dp):
                  dp[coin] = 1
  
          for x in range(1, amount+1):
              if dp[x]==1:
                  continue
  
              min_v = None
              for c in coins:
  
                  if x > c and dp[x-c] > 0:
                      if min_v==None:
                          min_v = dp[x-c] + 1
                      else:
                          min_v = min(min_v, dp[x-c]+1)
                  
              if min_v!=None:
                  dp[x] = min_v
  
          if dp[-1] > 0:
              return dp[-1]
          else:
              return -1
  
                  
  ```


### 61. 二叉树的层次遍历

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  from queue import Queue
  class Solution:
      def levelOrder(self, root: TreeNode) -> List[List[int]]:
          if root==None:
              return []
  
          ans = []
          q = Queue()
          q.put((root, 1))
          while not q.empty():
              font, i = q.get()
              if i > len(ans):
                  ans.append([font.val])
              else:
                  ans[i-1].append(font.val)
  
              if font.left:
                  q.put((font.left, i+1))
              if font.right:
                  q.put((font.right, i+1))
  
          return ans
  ```

* 递归方法(深度优先遍历)

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, d, ans):
          if root==None:
              return
          
          if d > len(ans):
              ans.append([root.val])
          else:
              ans[d-1].append(root.val)
  
          self.dfs(root.left, d+1, ans)
          self.dfs(root.right, d+1, ans)
  
      def levelOrder(self, root: TreeNode) -> List[List[int]]:
          ans = []
          self.dfs(root, 1, ans)
          return ans
              
  ```

### 62. 二叉树的锯齿形遍历(未完)

* 队列

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  from queue import Queue
  class Solution:
      def zigzagLevelOrder(self, root: TreeNode) -> List[List[int]]:
          if root==None:
              return []
  
          ans = []
          q = Queue()
          q.put((root, 1))
          while not q.empty():
              node, i = q.get()
              if i > len(ans):
                  ans.append([node.val])
              else:
                  if i%2==1:
                      ans[i-1].append(node.val)
                  else:
                      ans[i-1].insert(0, node.val)
              
              if node.left:
                  q.put((node.left, i+1))
              if node.right:
                  q.put((node.right, i+1))
  
          return ans
  ```

### 63. 整数拆分

* 动态规划

  ```python
  class Solution:
      def integerBreak(self, n: int) -> int:
  
          dp = [0 for _ in range(n+1)]
          dp[0] = 0
          dp[1] = 1
          for i in range(2, n+1):
              for j in range(1, i):
                  dp[i] = max(dp[i], dp[i-j]*j, (i-j)*j)
  
          return dp[-1]
  ```

* 公式法

  ```python
  class Solution:
      def integerBreak(self, n: int) -> int:
  
          if n <= 3:
              return n-1
  
          a = n//3
          b = n%3
          if b==0:
              return pow(3, a)
          if b==1:
              return pow(3, a-1)*4
          if b==2:
              return pow(3, a)*b
  ```

### 64. 最大整除子集

* 动态规划

  ```python
  class Solution:
      def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
          if len(nums)==0:
              return []
  
          nums.sort()
          ans = []
          dp = [1 for _ in range(len(nums))]
          for i in range(1, len(nums)):
              for j in range(i-1, -1, -1):
                  if nums[i]%nums[j]==0:
                      dp[i] = max(dp[i], dp[j]+1)
          
          count = max(dp)
          idx = dp.index(count)
          for j in range(idx, -1, -1):
              if nums[idx]%nums[j]==0 and dp[j]==count:
                  ans.append(nums[j])
                  count -= 1
  
          return ans
  ```

### 65. 组合总数

* 动态规划

  ```python
  class Solution:
      def combinationSum4(self, nums: List[int], target: int) -> int:
  
          dp = [0 for _ in range(target+1)]
          dp[0] = 1
          for x in range(1, target+1):
              for num in nums:
                  if x >= num:
                      dp[x] += dp[x-num]
  
          return dp[-1]
  ```


### 66. 等差数列划分

* 动态规划

  ```python
  class Solution:
      def numberOfArithmeticSlices(self, A: List[int]) -> int:
  
          dp = [[None for _ in range(len(A))] for _ in range(len(A))]
          count = 0
          for i in range(len(A)):
              for j in range(len(A)):
                  if j <= i+1:
                      dp[i][j] = False
                  elif j==i+2:
                      dp[i][j] = True if (A[j]-A[j-1]==A[j-1]-A[j-2]) else False
                  else:
                      dp[i][j] = True if (A[j]-A[j-1]==A[j-1]-A[j-2] and dp[i][j-1]==True) else False
                  if dp[i][j]==True:
                      count += 1
  
          return count
  ```

* 动态规划（空间优化）

  ```python
  class Solution:
      def numberOfArithmeticSlices(self, A: List[int]) -> int:
  
          dp = [None for _ in range(len(A))]
          count = 0
          for i in range(len(A)):
              for j in range(i, len(A)):
                  if j <= i+1:
                      dp[j] = False
                  elif A[j]-A[j-1]==A[j-1]-A[j-2]:
                      if j==i+2 or dp[j-1]==True:
                          dp[j] = True
                  else:
                      dp[j] = False
                  
                  if dp[j]==True:
                      count += 1
  
          return count
  ```

* 动态规划2

  ```python
  class Solution:
      def numberOfArithmeticSlices(self, A: List[int]) -> int:
          if len(A) < 3:
              return 0
  
          dp = [0 for _ in range(len(A))]
          dp[0] = 0
          dp[1] = 0
          for i in range(2, len(A)):
              if A[i]-A[i-1]==A[i-1]-A[i-2]:
                  dp[i] = 1 + dp[i-1]
          
          return sum(dp)
  ```

* 动态规划2优化

  ```python
  class Solution:
      def numberOfArithmeticSlices(self, A: List[int]) -> int:
          if len(A) < 3:
              return 0
  
          prev = 0
          sums = 0
          for i in range(2, len(A)):
              if A[i]-A[i-1]==A[i-1]-A[i-2]:
                  prev += 1
                  sums += prev
              else:
                  prev = 0
          
          return sums
  ```

  

### 67. 加油站（未完）

* 暴力

  ```python
  class Solution:
      def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
  
          drop = [gas[i]-cost[i] for i in range(len(gas))]
  
          for i in range(len(drop)):
              if drop[i] < 0:
                  continue
  
              j = (i + 1)%len(drop)
              sums = drop[i]
              while j!=i and sums >= 0:
                  sums += drop[j]
                  j = (j + 1)%len(drop)
              if sums >= 0:
                  return i
          
          return -1
  ```


### 68. 摆动序列(未完)

* 动态规划

  ```python
  class Solution:
      def wiggleMaxLength(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
  
          up = [1 for _ in range(len(nums))]
          down = [1 for _ in range(len(nums))]
          for i in range(1, len(nums)):
  
              for j in range(i-1, -1, -1):
                  if nums[j] < nums[i]:
                      up[i] = max(up[i], down[j] + 1)
                  elif nums[j] > nums[i]:
                      down[i] = max(down[i], up[j] + 1)
                      
          return max(max(up), max(down))
  ```

* 贪心算法

  ```python
  class Solution:
      def wiggleMaxLength(self, nums: List[int]) -> int:
          if len(nums)==0:
              return 0
          
          i = 0
          ans = [nums[0]]
          while i < len(nums)-1:
              j = i
              while i < len(nums)-1 and nums[i] <= nums[i+1]:
                  i += 1
              if i > j and nums[i]!=nums[j]:
                  ans.append(nums[i])
              
              j = i
              while i < len(nums)-1 and nums[i] >= nums[i+1]:
                  i += 1
              
              if i > j and nums[i]!=nums[j]:
                  ans.append(nums[i])
  
          return len(ans)
  ```

* 贪心算法O(1)

  ```python
  class Solution:
      def wiggleMaxLength(self, nums: List[int]) -> int:
          
          if len(nums) < 2:
              return len(nums)
  
          ans = 1
          i = 0
          while i < len(nums)-1:
              
              temp = i
              while i < len(nums)-1 and nums[i] <= nums[i+1]:
                  i += 1
  
              ans += nums[i] > nums[temp]
              
              temp = i 
              while i < len(nums)-1 and nums[i] >= nums[i+1]:
                  i += 1
  
              ans += nums[i] < nums[temp]
  
              
  
          return ans
  ```

  

### 69. 划分字母区间

* 贪心算法

  ```python
  class Solution:
      def partitionLabels(self, S: str) -> List[int]:
  
          last = {c:i for i, c in enumerate(S)}
  
          i = 0
          ans = []
          while i < len(S):
              j = last[S[i]]
              temp = i
              while i < j:
                  j = max(j, last[S[i]])
                  i += 1
              
              ans.append(i-temp+1)
              i += 1
  
          return ans
  ```

### 70. 单调递增的数字

* 逐个构造

  ```python
  class Solution:
      def monotoneIncreasingDigits(self, N: int) -> int:
  
          S = list(map(int, list(str(N))))
  
          i = 0
          while i < len(S)-1 and S[i] <= S[i+1]:
              i += 1
          
          if i >= len(S)-1:
              return N
          
          while i >= 0 and S[i] > S[i+1]:
              S[i] -= 1
              i -= 1
          
          for j in range(i+2, len(S)):
              S[j] = 9
  
          return int(''.join(map(str, S)))
  ```


### 71. 从前序与中序遍历序列构造二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
          if len(preorder)==0:
              return None
  
          node = TreeNode(preorder[0])
          idx = inorder.index(preorder[0])
  
          node.left = self.buildTree(preorder[1:idx+1], inorder[:idx])
          node.right = self.buildTree(preorder[idx+1:], inorder[idx+1:])
          return node
  
  ```

### 72. 从中序与后序遍历序列构造二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
      def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
          if len(inorder)==0:
              return None
          
          node = TreeNode(postorder[-1])
          idx = inorder.index(node.val)
  
          node.left = self.buildTree(inorder[:idx], postorder[:idx])
          node.right = self.buildTree(inorder[idx+1:], postorder[idx:-1])
  
          return node
  ```

  

### 73. 重复的DNA序列

* 集合

  ```python
  class Solution:
      def findRepeatedDnaSequences(self, s: str) -> List[str]:
          if len(s) <= 10:
              return []
  
          subs = set()
          ans = set()
          for i in range(10, len(s)+1):
              sv = s[i-10:i]
              if sv in subs:
                  ans.add(sv)
              else:
                  subs.add(sv)
          
          return list(ans)
  ```

  

### 74. 路径总和II

* 深度优先遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, sums, path, ans):
          if root==None:
              return
          
          if root.left==None and root.right==None and sums==root.val:
              path.append(root.val)
              ans.append(path[:])
              return
          
          path.append(root.val)
          self.dfs(root.left, sums-root.val, path[:], ans)
          self.dfs(root.right, sums-root.val, path[:], ans)
  
      def pathSum(self, root: TreeNode, sum: int) -> List[List[int]]:
          ans = []
          self.dfs(root, sum, [], ans)
          return ans
  ```


### 75. 岛屿的数量

* 深度优先遍历

  ```python
  from queue import Queue
  class Solution:
  
      def dfs(self, grid, r, c):
          dr = [1, 0, -1, 0]
          dc = [0, 1, 0, -1]
          for i in range(4):
              rr = r + dr[i]
              cc = c + dc[i]
              if rr >= 0 and rr < len(grid) and cc >= 0 and cc < len(grid[0]) \
              and grid[rr][cc]=='1':
                  grid[rr][cc] = '0'
                  self.dfs(grid, rr, cc)
  
  
      def numIslands(self, grid: List[List[str]]) -> int:
          
          count = 0
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]=='1':
                      self.dfs(grid, i, j)
                      count += 1
          
          return count
  ```

* 广度优先遍历

  ```python
  from queue import Queue
  class Solution:
  
      def numIslands(self, grid: List[List[str]]) -> int:
          
          count = 0
          q = Queue()
          direct = [(1, 0), (0, 1), (-1, 0), (0, -1)]
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]=='1':
                      count += 1
                      q.put((i, j))
                      grid[i][j] = '0'
                      while not q.empty():
                          r, c = q.get()
                          for dr, dc in direct:
                              rr = r + dr
                              cc = c + dc
                              if rr >= 0 and rr < len(grid) and cc >= 0 and cc < len(grid[0]) and grid[rr][cc]=='1':
                                  grid[rr][cc] = '0'
                                  q.put((rr, cc))
  
          
          return count
  ```

* 访问矩阵

  ```python
  from queue import Queue
  class Solution:
  
      def numIslands(self, grid: List[List[str]]) -> int:
          
          count = 0
          q = Queue()
          direct = [(1, 0), (0, 1), (-1, 0), (0, -1)]
          visit = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  if grid[i][j]=='1' and visit[i][j]==False:
                      count += 1
                      q.put((i, j))
                      visit[i][j] = True
                      while not q.empty():
                          r, c = q.get()
                          for dr, dc in direct:
                              rr = r + dr
                              cc = c + dc
                              if rr >= 0 and rr < len(grid) and cc >= 0 and cc < len(grid[0]) and grid[rr][cc]=='1' and visit[rr][cc]==False:
                                  q.put((rr, cc))
                                  visit[rr][cc] = True
  
          
          return count
  ```

  

### 76. 求根到叶子节点数字之和

* 深度优先遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, path):
          if root==None:
              return
  
          path = path*10 + root.val
          if root.left==None and root.right==None:
              self.ans += path
              return
          self.dfs(root.left, path)
          self.dfs(root.right, path)
          
  
  
      def sumNumbers(self, root: TreeNode) -> int:
  
          self.ans = 0
          self.dfs(root, 0)
          return self.ans
  ```

* 深度优先遍历(更耗时)

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, path):
          if root==None:
              return 0
  
          path = path*10 + root.val
          if root.left==None and root.right==None:
              return path
          return self.dfs(root.left, path) + self.dfs(root.right, path)
          
  
  
      def sumNumbers(self, root: TreeNode) -> int:
  
          return self.dfs(root, 0)
  ```

### 77. 二叉树的右视图

* 深度优先遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, depth, ans):
          if root==None:
              return
          
          if depth > len(ans):
              ans.append(root.val)
          self.dfs(root.right, depth+1, ans)
          self.dfs(root.left, depth+1, ans)
  
      def rightSideView(self, root: TreeNode) -> List[int]:
  
          ans = []
          self.dfs(root, 1, ans)
          return ans
  ```

* 广度优先遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  from queue import Queue
  class Solution:
  
      def rightSideView(self, root: TreeNode) -> List[int]:
          if root==None:
              return []
  
          ans = []
          q = Queue()
          q.put((root, 1))
          while not q.empty():
              node, d = q.get()
              if node.right:
                  q.put((node.right, d+1))
              if node.left:
                  q.put((node.left, d+1))
              
              if d > len(ans):
                  ans.append(node.val)
  
          return ans
  ```

### 78. 数字范围按位与

* 暴力

  ```python
  class Solution:
      def rangeBitwiseAnd(self, m: int, n: int) -> int:
  
          left = m
          right = n
          ans = []
          while left <= right and right > 0:
  
              count = 0
              for x in range(left, right+1):
                  if x%2==0:
                      count += 1
                      break
              if count==0:
                  ans.append(1)
              else:
                  ans.append(0)
              left //= 2
              right //= 2
  
          val = 0
          for v in ans[::-1]:
              val = val*2 + v
  
          return val
  
  ```

### 79. 实现Trie

* 字典

  ```python
  class Trie:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.tree = {}
  
  
      def insert(self, word: str) -> None:
          """
          Inserts a word into the trie.
          """
          node = self.tree
          for w in word:
              if w not in node:
                  node[w] = {}
              node = node[w]
          node['#'] = '#'
  
  
      def search(self, word: str) -> bool:
          """
          Returns if the word is in the trie.
          """
          node = self.tree
          for w in word:
              if w not in node:
                  return False
              node = node[w]
  
          if '#' not in node:
              return False
          
          return True
  
  
      def startsWith(self, prefix: str) -> bool:
          """
          Returns if there is any word in the trie that starts with the given prefix.
          """
          node = self.tree
          for p in prefix:
              if p not in node:
                  return False
              node = node[p]
          return True
  
  
  
  
  # Your Trie object will be instantiated and called as such:
  # obj = Trie()
  # obj.insert(word)
  # param_2 = obj.search(word)
  # param_3 = obj.startsWith(prefix)
  ```

* 定义结点类

  ```python
  from collections import defaultdict
  class TrieNode:
      def __init__(self):
          self.children = defaultdict(TrieNode)
          self.isword = False
  
  
  class Trie:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.tree = TrieNode()
  
  
      def insert(self, word: str) -> None:
          """
          Inserts a word into the trie.
          """
          node = self.tree
          for w in word:
              node = node.children[w]
          node.isword = True
  
  
      def search(self, word: str) -> bool:
          """
          Returns if the word is in the trie.
          """
          node = self.tree
          for w in word:
              if w not in node.children:
                  return False
              node = node.children[w]
  
          return node.isword
  
  
      def startsWith(self, prefix: str) -> bool:
          """
          Returns if there is any word in the trie that starts with the given prefix.
          """
          node = self.tree
          for p in prefix:
              if p not in node.children:
                  return False
              node = node.children[p]
          return True
  
  
  
  
  # Your Trie object will be instantiated and called as such:
  # obj = Trie()
  # obj.insert(word)
  # param_2 = obj.search(word)
  # param_3 = obj.startsWith(prefix)
  ```

  

### 80. 长度最小的子数组

* 动态规划

  ```python
  class Solution:
      def minSubArrayLen(self, s: int, nums: List[int]) -> int:
          if s==0 or len(nums)==0:
              return 0
          
          dp = [[0, 0] for _ in range(len(nums))]
          dp[0][0] = nums[0]
  
          for i in range(1, len(nums)):
              dp[i][0] = dp[i-1][0] + nums[i]
              j = dp[i-1][1]
              while dp[i][0]-nums[j] >= s:
                  dp[i][0] -= nums[j]
                  j += 1
              dp[i][1] = j
          
          min_len = len(nums) + 1
          for i in range(len(dp)):
              if dp[i][0] >= s:
                  min_len = min(i-dp[i][1]+1, min_len)
          if min_len <= len(nums):
              return min_len
          else:
              return 0
  ```

* 动态规划2

  ```python
  class Solution:
      def minSubArrayLen(self, s: int, nums: List[int]) -> int:
          if s==0 or len(nums)==0:
              return 0
          
          dp = [[0, 0] for _ in range(len(nums))]
          dp[0][0] = nums[0]
  
          min_len = len(nums) + 1
          if dp[0][0] >= s:
              min_len = min(1, min_len)
          
          for i in range(1, len(nums)):
              dp[i][0] = dp[i-1][0] + nums[i]
              j = dp[i-1][1]
              while dp[i][0]-nums[j] >= s:
                  dp[i][0] -= nums[j]
                  j += 1
              dp[i][1] = j
  
              if dp[i][0] >= s:
                  min_len = min(i-dp[i][1]+1, min_len)
          
          if min_len <= len(nums):
              return min_len
          else:
              return 0
  ```

* 动态规划优化

  ```python
  class Solution:
      def minSubArrayLen(self, s: int, nums: List[int]) -> int:
          if s==0 or len(nums)==0:
              return 0
          
          left = 0
          sums = nums[0]
          if sums >= s:
              return 1
  
          ans = len(nums) + 1
          for i in range(1, len(nums)):
              sums += nums[i]
              while sums-nums[left] >= s:
                  sums -= nums[left]
                  left += 1
              
              if sums >= s:
                  ans = min(ans, i-left+1)
  
          if ans <= len(nums):
              return ans
          else:
              return 0
  ```


### 81. 被围绕的区域

* 深度优先遍历

  ```python
  class Solution:
  
      def dfs(self, board, i, j):
          board[i][j] = None
          if i+1 < len(board) and board[i+1][j]=='O':
              self.dfs(board, i+1, j)
  
          if j+1 < len(board[0]) and board[i][j+1]=='O':
              self.dfs(board, i, j+1)
  
          if i-1 >= 0 and board[i-1][j]=='O':
              self.dfs(board, i-1, j)
  
          if j-1 >= 0 and board[i][j-1]=='O':
              self.dfs(board, i, j-1)
  
  
      def solve(self, board: List[List[str]]) -> None:
          """
          Do not return anything, modify board in-place instead.
          """
          if len(board)==0:
              return board
          for j in range(len(board[0])):
              if board[0][j]=='O':
                  self.dfs(board, 0, j)
              if board[-1][j]=='O':
                  self.dfs(board, len(board)-1, j)
          for i in range(len(board)):
              if board[i][-1]=='O':
                  self.dfs(board, i, len(board[0])-1)
              if board[i][0]=='O':
                  self.dfs(board, i, 0)
  
          for i in range(len(board)):
              for j in range(len(board[0])):
                  if board[i][j]=='O':
                      board[i][j] = 'X'
                  if board[i][j]==None:
                      board[i][j] = 'O'
  ```


### 82. 寻找重复数

* 集合

  ```python
  class Solution:
      def findDuplicate(self, nums: List[int]) -> int:
  
          visit = set()
          for num in nums:
              if num in visit:
                  return num
              visit.add(num)
  ```

* 二分法

  ```python
  class Solution:
      def findDuplicate(self, nums: List[int]) -> int:
  
          left = 1
          right = len(nums)-1
  
          while left < right:
              mid = (left + right)//2
              count = 0
              for i in range(len(nums)):
                  if nums[i] <= mid:
                      count += 1
              if count > mid:
                  right = mid
              else:
                  left = mid + 1
  
          return right
  ```

### 83. 逆波兰式

* 栈(python整除向下取整)

  ```python
  class Solution:
      def evalRPN(self, tokens: List[str]) -> int:
  
          stk = []
          for t in tokens:
              if t in ['+', '-', '*', '/']:
                  y = stk.pop()
                  x = stk.pop()
                  if t=='+':
                      stk.append(x+y)
                  if t=='-':
                      stk.append(x-y)
                  if t=='*':
                      stk.append(x*y)
                  if t=='/':
                      if x%y==0 or (x*y) >= 0:
                          stk.append(x//y)
                      else:
                          stk.append(x//y+1)
              else:
                  stk.append(int(t))
          return stk[-1]
  
  ```


### 84. 灯泡开关

* 暴力法（超时）

  ```python
  class Solution:
      def bulbSwitch(self, n: int) -> int:
          nums = [0 for _ in range(n)]
  
          for i in range(1, n+1):
              j = i-1
              while j < n:
                  nums[j] = 1 - nums[j]
                  j += i
  
          return sum(nums)
  ```

* 统计平方数的个数

  ```python
  import math
  class Solution:
  
      def bulbSwitch(self, n: int) -> int:
          return int(math.sqrt(n))
  ```

  

### 85. 翻转字符串里的单词

* 暴力

  ```python
  class Solution:
      def reverseWords(self, s: str) -> str:
          ans = []
          i = len(s)-1
          while i >= 0:
              while i >= 0 and s[i]==' ':
                  i -= 1
  
              j = i
              while j >= 0 and s[j]!=' ':
                  j -= 1
              
  
              if j < i:
                  ans.append(s[j+1:i+1])
              
              i = j
  
          return ' '.join(ans)
  
  ```

* reversed + split

  ```python
  class Solution:
      def reverseWords(self, s: str) -> str:
          ans = s.split()
          return ' '.join(reversed(ans))
  ```

### 86. 寻找排序数组中的最小值

* 线性扫描

  ```python
  class Solution:
      def findMin(self, nums: List[int]) -> int:
          if len(nums) < 3:
              return min(nums)
  
          for i in range(1, len(nums)-1):
              if nums[i] < nums[i-1] and nums[i] < nums[i+1]:
                  return nums[i]
          
          return min(nums[0], nums[-1])
  ```

* 二分法

  ```python
  class Solution:
      def findMin(self, nums: List[int]) -> int:
          left = 0
          right = len(nums)-1
          minv = nums[0]
  
          while left < right:
              mid = (left + right)//2
              if nums[mid] < nums[right]:
                  minv = min(minv, nums[mid])
                  right = mid - 1
              else:
                  minv = min(minv, nums[left])
                  left = mid + 1
          return min(minv, nums[left])
  ```

### 87. 寻找峰值

* 线性扫描

  ```python
  class Solution:
      def findPeakElement(self, nums: List[int]) -> int:
          if len(nums)==1:
              return 0
  
          if nums[0] > nums[1]:
              return 0
          if nums[-1] > nums[-2]:
              return len(nums)-1
          
          for i in range(1, len(nums)-1):
              if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                  return i
  ```

* 二分法

  ```python
  class Solution:
      def findPeakElement(self, nums: List[int]) -> int:
          left = 0
          right = len(nums)-1
          while left < right:
              mid = (left + right)//2
              if nums[mid] > nums[mid+1]:
                  right = mid
              else:
                  left = mid + 1
  
          return left
  ```

  

### 88. 比较版本号

* 分割

  ```python
  class Solution:
      def compareVersion(self, version1: str, version2: str) -> int:
          v1 = list(map(int, version1.split('.')))
          v2 = list(map(int, version2.split('.')))
  
          for i in range(max(len(v1), len(v2))):
              if i >= len(v1):
                  if v2[i] > 0:
                      return -1
                  if v2[i] < 0:
                      return 1
              elif i >= len(v2):
                  if v1[i] > 0:
                      return 1
                  if v1[i] < 0:
                      return -1
              else:
                  if v1[i] > v2[i]:
                      return 1
                  if v1[i] < v2[i]:
                      return -1
  
          return 0
  ```

### 89. 二叉搜索树迭代器

* 中序遍历

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class BSTIterator:
  
      def __init__(self, root: TreeNode):
          self.root = root
          self.nums = []
          self.dfs(self.root, self.nums)
          self.index = 0
  
      def dfs(self, root, nums):
          if root==None:
              return
  
          self.dfs(root.left, nums)
          nums.append(root.val)
          self.dfs(root.right, nums)
  
      def next(self) -> int:
          """
          @return the next smallest number
          """
          if self.index < len(self.nums):
              self.index += 1
              return self.nums[self.index-1]
              
  
      def hasNext(self) -> bool:
          """
          @return whether we have a next smallest number
          """
          return self.index < len(self.nums)
          
  ```

* 受控迭代

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class BSTIterator:
  
      def __init__(self, root: TreeNode):
          self.stk = []
          self.dfs(root)
  
  
      def dfs(self, root):
          while root:
              self.stk.append(root)
              root = root.left
  
      def next(self) -> int:
          """
          @return the next smallest number
          """
          top = self.stk.pop()
          if top.right:
              self.dfs(top.right)
          return top.val
              
  
      def hasNext(self) -> bool:
          """
          @return whether we have a next smallest number
          """
          return len(self.stk) > 0
          
  ```

  


### 90. 最大单词长度乘积

* 暴力(超时)

  ```python
  class Solution:
  
      def not_common_letters(self, w1, w2):
          s1 = set(w1)
          s2 = set(w2)
          return len(s1 & s2)==0
  
      def maxProduct(self, words: List[str]) -> int:
  
          ans = 0
  
          for i in range(len(words)):
              for j in range(len(words)):
                  if i==j:
                      continue
  
                  if self.not_common_letters(words[i], words[j]):
                      ans = max(ans, len(words[i])*len(words[j]))
  
          return ans
  ```

* 优化no_common_letters(超时)

  ```python
  class Solution:
  
      # 单词掩码
      def no_common_letters(self, s1, s2):
          bit_number = lambda ch : ord(ch) - ord('a')
  
          bitmask1 = bitmask2 = 0
          for ch in s1:
              bitmask1 |= 1 << bit_number(ch)
          for ch in s2:
              bitmask2 |= 1 << bit_number(ch)
          return bitmask1 & bitmask2 == 0
  
  
      def maxProduct(self, words: List[str]) -> int:
  
          ans = 0
  
          for i in range(len(words)):
              for j in range(len(words)):
                  if i==j:
                      continue
  
                  if self.no_common_letters(words[i], words[j]):
                      ans = max(ans, len(words[i])*len(words[j]))
  
          return ans
  ```

* 哈希+单词掩码

  ```python
  from collections import defaultdict
  class Solution:
  
  
      def maxProduct(self, words: List[str]) -> int:
          pos = lambda x: ord(x)-ord('a')
  
          masks = defaultdict(int)
          for word in words:
              msk = 0
              for w in word:
                  msk |= 1 << pos(w)
              masks[msk] = max(masks[msk], len(word))
  
          ans = 0
          for k1, v1 in masks.items():
              for k2, v2 in masks.items():
                  if k1&k2==0:
                      ans = max(ans, v1*v2)
          return ans
  ```

### 91. 奇偶链表

* 暴力

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  class Solution:
      def oddEvenList(self, head: ListNode) -> ListNode:
          if head==None or head.next==None:
              return head
          
          odd = head
          even = head.next
          p = odd
          q = even
          while p.next and q.next:
              if q.next:
                  p.next = q.next
                  p = p.next
              if p.next:
                  q.next = p.next
                  q = q.next
          p.next = even
          q.next = None
          return odd
  ```


### 92. 比特位计数 
* 暴力
  ```python
  
  ```

class Solution:

      def countBits(self, num: int) -> List[int]:
      
          ans = []
          for x in range(0, num+1):
              
              count = 0
              while x:
                  count += x&1
                  x >>= 1
              ans.append(count)
      
          return ans

  













  ```

* 动态规划1

  ```python
  class Solution:
  
      def countBits(self, num: int) -> List[int]:
          if num==0:
              return [0]
          dp = [0 for _ in range(num+1)]
          dp[1] = 1
          b = 1
          for i in range(2, num+1):
              if i%b==0:
                  b <<= 1
              dp[i] = dp[i-b] + 1
  
          return dp
  ```

* 动态规划2

  ```python
  class Solution:
  
      def countBits(self, num: int) -> List[int]:
          if num==0:
              return [0]
          dp = [0 for _ in range(num+1)]
          dp[1] = 1
          b = 1
          for i in range(2, num+1):
              dp[i] = dp[i//2] + (i%2)
  
          return dp
  ```

* 动态规划（清除最后一个1）

  ```python
  class Solution:
  
      def countBits(self, num: int) -> List[int]:
          if num==0:
              return [0]
          dp = [0 for _ in range(num+1)]
          dp[1] = 1
          b = 1
          for i in range(2, num+1):
              dp[i] = dp[i&(i-1)] + 1
  
          return dp
  ```

  

### 93. 前K个高频元素

* 堆排序

  ```python
  class Solution:
  
      def adjust(self, nums, i, length):
          max_idx = i
          left = 2*i + 1
          right = 2*i + 2
          if left < length and nums[left][1] > nums[max_idx][1]:
              max_idx = left
          if right < length and nums[right][1] > nums[max_idx][1]:
              max_idx = right
          
          if max_idx!=i:
              temp = nums[i]
              nums[i] = nums[max_idx]
              nums[max_idx] = temp
              self.adjust(nums, max_idx, length)
  
  
      def topKFrequent(self, nums: List[int], k: int) -> List[int]:
  
          ans = []
          counts = {}
          for num in nums:
              if num in counts:
                  counts[num] += 1
              else:
                  counts[num] = 1
          
          counts = list(zip(counts.keys(), counts.values()))
          for i in range(len(counts)//2-1, -1, -1):
              self.adjust(counts, i, len(counts))
          
          for j in range(1, k+1):
              ans.append(counts[0][0])
              temp = counts[0]
              counts[0] = counts[len(counts)-j]
              counts[len(counts)-j]
              self.adjust(counts, 0, len(counts)+1-j)
  
          return ans
  ```

  

### 94. 字典序排数

* dfs

  ```python
  class Solution:
  
      def dfs(self, i, n, ans):
          if i > n:
              return
          ans.append(i)
          for e in range(10):
              j = 10*i + e
              self.dfs(j, n, ans)
  
      def lexicalOrder(self, n: int) -> List[int]:
          ans = []
          for i in range(1, 10):
              self.dfs(i, n, ans)
          return ans
  ```

### 95. 旋转函数

* 错位相减

  ```python
  class Solution:
      def maxRotateFunction(self, A: List[int]) -> int:
          f = 0
          for i, v in enumerate(A):
              f += i*v
  
          ans = f
          sums = sum(A)
          i = len(A)-1
  
          for _ in range(1, len(A)):
              f += sums-len(A)*A[i]
              ans = max(ans, f)
              i -= 1
          
          return ans
  ```

### 96. 常数时间插入删除和获取元素

* 哈希和动态数组

  ```python
  from random import choice
  class RandomizedSet:
  
      def __init__(self):
          """
          Initialize your data structure here.
          """
          self.array = []
          self.index = {}
          
  
      def insert(self, val: int) -> bool:
          """
          Inserts a value to the set. Returns true if the set did not already contain the specified element.
          """
          if val in self.index:
              return False
  
          self.index[val] = len(self.array)
          self.array.append(val)
          return True
          
  
      def remove(self, val: int) -> bool:
          """
          Removes a value from the set. Returns true if the set contained the specified element.
          """
          if val not in self.index:
              return False
          
          idx = self.index[val]
          self.array[idx] = self.array[-1]
          self.index[self.array[-1]] = idx
          del self.index[val]
          self.array.pop()
          return True
          
  
      def getRandom(self) -> int:
          """
          Get a random element from the set.
          """
          return choice(self.array)
  ```

### 97. 整数替换

* 递归

  ```python
  class Solution:
  
      def dfs(self, n):
          if n==1:
              return 0
          
          if n%2==0:
              return self.dfs(n//2) + 1
          else:
              v1 = 1 + self.dfs(n-1)
              v2 = 2 + self.dfs((n+1)//2)
              return min(v1, v2)
  
  
      def integerReplacement(self, n: int) -> int:
  
          return self.dfs(n)
  ```

  

### 98. 消除游戏

* 模拟（超时）

  ```python
  class Solution:
      def lastRemaining(self, n: int) -> int:
          nums = [i+1 for i in range(n)]
          while len(nums) > 1:
              i = 0
              while len(nums) > 1 and i < len(nums):
                  nums.pop(i)
                  i += 1
  
              i = len(nums)-1
              while len(nums) > 1 and i >= 0:
                  nums.pop(i)
                  i -= 2
          
          return nums[0]
  ```

* 非删除模拟（超时）

  ```python
  class Solution:
      def lastRemaining(self, n: int) -> int:
          nums = [i+1 for i in range(n)]
          count = len(nums)
          while count > 1:
  
              c = 0
              for i in range(len(nums)):
                  if nums[i]==0:
                      continue
                  if count==1:
                      break
  
                  if c==0:
                      nums[i] = 0
                      count -= 1
                  c = (c+1)%2
              
              c = 0
              for i in range(len(nums)-1, -1, -1):
                  if nums[i]==0:
                      continue
                  if count==1:
                      break
  
                  if c==0:
                      nums[i] = 0
                      count -= 1
                  c = (c+1)%2
  
          return sum(nums)
  ```

* 规律

  ```python
  class Solution:
      def lastRemaining(self, n: int) -> int:
          first = 1
          diff = 1
          fromleft = True
          count = n
          while count > 1:
              if fromleft:
                  first += diff
              else:
                  if count%2==1:
                      first += diff
              
              count >>= 1
              diff <<= 1
              fromleft = not fromleft
          
          return first
  ```

  

### 99. 有序矩阵中第K小的元素

* 二分法

  ```python
  class Solution:
  
      def count_less(self, matrix, mid, row, col):
          i = row-1
          j = 0
          count = 0
          while i >= 0 and j < col:
              if matrix[i][j] <= mid:
                  count += i+1
                  j += 1
              else:
                  i -= 1
          return count
  
  
      def kthSmallest(self, matrix: List[List[int]], k: int) -> int:
          left = matrix[0][0]
          right = matrix[-1][-1]
          row, col = len(matrix), len(matrix[0])
          while left < right:
              mid = (left + right)//2
              count = self.count_less(matrix, mid, row, col)
              if count < k:
                  left = mid + 1
              else:
                  right = mid
  
          return left
  
  
  ```

  

### 100. 随机数索引

* 线性扫描

  ```python
  class Solution:
      def lastRemaining(self, n: int) -> int:
          first = 1
          diff = 1
          fromleft = True
          count = n
          while count > 1:
              if fromleft:
                  first += diff
              else:
                  if count%2==1:
                      first += diff
              
              count >>= 1
              diff <<= 1
              fromleft = not fromleft
          
          return first
  ```



### 101. 甲板上的战舰

* dfs

  ```python
  class Solution:
  
      def dfs(self, board, i, j):
          if board[i][j]=='X':
              board[i][j] = '.'
          
          if i-1 >= 0 and board[i-1][j]=='X':
              self.dfs(board, i-1, j)
          if j-1 >= 0 and board[i][j-1]=='X':
              self.dfs(board, i, j-1)
          if i+1 < len(board) and board[i+1][j]=='X':
              self.dfs(board, i+1, j)
          if j+1 < len(board[0]) and board[i][j+1]=='X':
              self.dfs(board, i, j+1)
  
      def countBattleships(self, board: List[List[str]]) -> int:
  
          count = 0
          for i in range(len(board)):
              for j in range(len(board[0])):
                  if board[i][j]=='X':
                      self.dfs(board, i, j)
                      count += 1
          return count
  ```

* 线性扫描

  ```python
  class Solution:
  
  
      def countBattleships(self, board: List[List[str]]) -> int:
          
          ans = 0
          for i in range(len(board)):
              for j in range(len(board[0])):
                  if board[i][j]=='X':
                      if (i==0 or board[i-1][j]!='X') and (j==0 or board[i][j-1]!='X'):
                          ans += 1
  
          return ans
  ```

  

### 102. 从英文中重建数字

* 计数法

  ```python
  from collections import defaultdict
  class Solution:
      def originalDigits(self, s: str) -> str:
          counts = defaultdict(int)
          for x in s:
              counts[x] += 1
          
          nums = [0 for _ in range(10)]
          nums[0] = counts['z']
          nums[2] = counts['w']
          nums[4] = counts['u']
          nums[6] = counts['x']
          nums[8] = counts['g']
          nums[1] = counts['o']-nums[0]-nums[2]-nums[4]
          nums[3] = counts['h']-nums[8]
          nums[7] = counts['s']-nums[6]
          nums[5] = counts['v']-nums[7]
          nums[9] = counts['i']-nums[5]-nums[6]-nums[8]
  
          ans = [str(i)*nums[i] for i in range(10) if nums[i] > 0]
          return ''.join(ans)
  ```


### 103. 打乱数组

* 暴力法

  时间复杂度为$O(n^2)$, 平方来自于list.pop，空间复杂度为O(n)。保存数组为temp，从temp中随机取元素，按顺序放入nums中，从temp中删除，直至temp为空

  ```python
  import random
  class Solution:
  
      def __init__(self, nums: List[int]):
          self.init = nums[:]
          self.nums = self.init[:]
          
  
      def reset(self) -> List[int]:
          """
          Resets the array to its original configuration and return it.
          """
          self.nums = self.init[:]
          return self.nums
          
  
      def shuffle(self) -> List[int]:
          """
          Returns a random shuffling of the array.
          """
          temp = self.nums[:]
          for i in range(len(self.nums)):
              j = random.randint(0, len(temp)-1)
              self.nums[i] = temp[j]
              temp.pop(j)
          return self.nums
  ```

* Fisher-Yates

  时间复杂度和空间复杂度都是$O(n)$, 采取的思想是交换。

  ```python
  import random
  class Solution:
  
      def __init__(self, nums: List[int]):
          self.init = nums[:]
          self.nums = self.init[:]
          
  
      def reset(self) -> List[int]:
          """
          Resets the array to its original configuration and return it.
          """
          self.nums = self.init[:]
          return self.nums
          
  
      def shuffle(self) -> List[int]:
          """
          Returns a random shuffling of the array.
          """
          for i in range(len(self.nums)):
              j = random.randint(i, len(self.nums)-1)
              temp = self.nums[i]
              self.nums[i] = self.nums[j]
              self.nums[j] = temp
          return self.nums
  ```

### 104. 链表随机节点

* 暴力法

  计算链表长度，随机生成长度范围内的索引，返回元素。

  ```python
  import random
  class Solution:
  
      def __init__(self, head: ListNode):
          """
          @param head The linked list's head.
          Note that the head is guaranteed to be not null, so it contains at least one node.
          """
          self.head = head
  
          self.length = 0
          p = head
          while p:
              p = p.next
              self.length += 1
          
  
      def getRandom(self) -> int:
          """
          Returns a random node's value.
          """
          i = random.randint(0, self.length-1)
          p = self.head
          while i > 0:
              p = p.next
              i -= 1
  
          return p.val
  ```

* 蓄水池采样算法

  从数据流中随机选取k个元素，保证每个元素被取到的概率相同

  ```python
  # Definition for singly-linked list.
  # class ListNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.next = None
  
  import random
  class Solution:
  
      def __init__(self, head: ListNode):
          """
          @param head The linked list's head.
          Note that the head is guaranteed to be not null, so it contains at least one node.
          """
          self.head = head
  
      def getRandom(self) -> int:
          """
          Returns a random node's value.
          """
          p = self.head
          val = None
          i = 1
          while p:
              j = random.randint(1, i)
              if j==i:
                  val = p.val
              i += 1
              p = p.next
  
          return val
  ```

  

### 105. N叉树的层序遍历

* 队列

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  from queue import Queue
  class Solution:
      def levelOrder(self, root: 'Node') -> List[List[int]]:
          if root==None:
              return []
  
          ans = []
          q = Queue()
          q.put((root, 0))
          while q.qsize() > 0:
              node, d = q.get()
              if d >= len(ans):
                  ans.append([node.val])
              else:
                  ans[d].append(node.val)
              
              for child in node.children:
                  q.put((child, d+1))
  
          return ans
  ```

* 深度优先搜索

  ```python
  """
  # Definition for a Node.
  class Node:
      def __init__(self, val=None, children=None):
          self.val = val
          self.children = children
  """
  from queue import Queue
  class Solution:
  
      def dfs(self, root, d, ans):
          if root==None:
              return
          if d >= len(ans):
              ans.append([root.val])
          else:
              ans[d].append(root.val)
  
          for child in root.children:
              self.dfs(child, d+1, ans)
  
      def levelOrder(self, root: 'Node') -> List[List[int]]:
          if root==None:
              return []
  
          ans = []
          self.dfs(root, 0, ans)
          return ans
  ```

### 106. 数组中重复的元素

* 集合

  使用额外空间

  ```python
  class Solution:
      def findDuplicates(self, nums: List[int]) -> List[int]:
  
          ans = []
          record = set()
          for num in nums:
              if num in record:
                  ans.append(num)
              else:
                  record.add(num)
  
          return ans
  ```

* 桶排序

  ```python
  class Solution:
      def findDuplicates(self, nums: List[int]) -> List[int]:
  
          i = 0
          while i < len(nums):
              if nums[i]==i+1 or nums[nums[i]-1]==nums[i]:
                  i += 1
                  continue
  
              j = nums[i]
              if nums[j-1]!=j:
                  nums[i] = nums[j-1]
                  nums[j-1] = j
  
          count = 0
          ans = []
          for i in range(len(nums)):
              if nums[i]!=i+1:
                  ans.append(nums[i])
  
          return ans
  ```

### 107. 找树左下角的值

* 深度优先

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, d, ans):
          if root==None:
              return
  
          if d not in ans:
              ans.append(d)
              self.val = root.val
          
          self.dfs(root.left, d+1, ans)
          self.dfs(root.right, d+1, ans)
  
      def findBottomLeftValue(self, root: TreeNode) -> int:
  
          ans = []
          self.val = None
          self.dfs(root, 0, ans)
          return self.val
  ```


### 108. 出现次数最多的子树元素和

* 计数法 + 深度优先

  ```python
  from collections import defaultdict
  class Solution:
  
      def dfs(self, root, counts):
          if root==None:
              return 0
  
          lc = self.dfs(root.left, counts)
          rc = self.dfs(root.right, counts)
          c = lc + rc + root.val
          counts[c] += 1
          return c
  
          
  
      def findFrequentTreeSum(self, root: TreeNode) -> List[int]:
          if root==None:
              return []
  
          counts = defaultdict(int)
          self.dfs(root, counts)
  
          ans = []
          maxc = max(counts.values())
          for k, v in counts.items():
              if v==maxc:
                  ans.append(k)
  
          return ans
  ```



### 109. 在每个树行中找最大值

* 深度优先

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def dfs(self, root, d, ans):
          if root==None:
              return
          
          if d >= len(ans):
              ans.append(root.val)
          else:
              ans[d] = max(ans[d], root.val)
  
          self.dfs(root.left, d+1, ans)
          self.dfs(root.right, d+1, ans)
  
      def largestValues(self, root: TreeNode) -> List[int]:
          ans = []
          self.dfs(root, 0, ans)
          return ans
  ```

* 队列（稍慢）

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  from queue import Queue
  class Solution:
  
      
      def largestValues(self, root: TreeNode) -> List[int]:
          if root==None:
              return []
  
          ans = []
          q = Queue()
          q.put((root, 0))
          while q.qsize() > 0:
              node, d = q.get()
              if d >= len(ans):
                  ans.append(node.val)
              else:
                  ans[d] = max(ans[d], node.val)
  
              if node.left:
                  q.put((node.left, d+1))
              if node.right:
                  q.put((node.right, d+1))
  
          return ans
  ```

### 110. 有序数组的单一元素

* 异或

  ```python
  class Solution:
      def singleNonDuplicate(self, nums: List[int]) -> int:
  
          ans = 0
          for num in nums:
              ans ^= num
          
          return ans
  ```

* 二分法

  ```python
  class Solution:
      def singleNonDuplicate(self, nums: List[int]) -> int:
  
          left = 0
          right = len(nums)-1
          while left < right:
              mid = (left + right)//2
              if nums[mid]==nums[mid-1]:
                  if (right-mid)%2==0:
                      right = mid-2
                  else:
                      left = mid + 1
              elif nums[mid]==nums[mid+1]:
                  if (mid-left)%2==0:
                      left = mid+2
                  else:
                      right = mid - 1
              else:
                  return nums[mid]
  
          return nums[right]
      
  ```

### 111. TinyURL 的加密与解密

* 计数法

  ```python
  class Codec:
  
      hashmap = {}
      i = 0
  
      def encode(self, longUrl: str) -> str:
          """Encodes a URL to a shortened URL.
          """
          self.hashmap[self.i] = longUrl
          url = "http://tinyurl.com/{}".format(self.i)
          self.i += 1
          return url
  
  
          
  
      def decode(self, shortUrl: str) -> str:
          """Decodes a shortened URL to its original URL.
          """
          return self.hashmap[int(shortUrl.replace("http://tinyurl.com/", ""))]
  ```

### 112. 朋友圈

* 深度优先搜索

  ```python
  class Solution:
  
      def dfs(self, M, i, j):
          M[i][j] = 0
          for k in range(len(M[0])):
              if M[j][k]==1:
                  self.dfs(M, j, k)
  
      def findCircleNum(self, M: List[List[int]]) -> int:
          
          count = 0
  
          for i in range(len(M)):
              for j in range(len(M[0])):
                  if M[i][j]==1:
                      self.dfs(M, i, j)
                      count += 1
  
          return count
  ```

* 广度优先搜

* 并查集


### 113. 根据前序和后续构建二叉树

* 递归

  ```python
  # Definition for a binary tree node.
  # class TreeNode:
  #     def __init__(self, x):
  #         self.val = x
  #         self.left = None
  #         self.right = None
  
  class Solution:
  
      def construct(self, root, pre, post):
          if root==None or len(pre)==0:
              return 
  
          root.val = pre[0]
          
          if len(pre)==1:
              return
  
          i = 0
          while i < len(post) and pre[1]!=post[i]:
              i += 1
  
          if len(pre[1:i+2]) > 0:
              root.left = TreeNode(0)
          if len(pre[i+2:]) > 0:
              root.right = TreeNode(0)
  
          self.construct(root.left, pre[1:i+2], post[:i+1])
          self.construct(root.right, pre[i+2:], post[i+1:-1])
              
  
  
      def constructFromPrePost(self, pre: List[int], post: List[int]) -> TreeNode:
  
          root = TreeNode(0)
          self.construct(root, pre, post)
          return root
  ```



### 114. 查找和替换模式

* 双映射表

  ```python
  class Solution:
  
      def ispattern(self, word, pattern):
          if len(word)!=len(pattern):
              return False
  
          leftmap = {}
          rightmap = {}
          for i in range(len(pattern)):
              if pattern[i] not in leftmap and word[i] not in rightmap:
                  leftmap[pattern[i]] = word[i]
                  rightmap[word[i]] = pattern[i]
                  continue
              if pattern[i] in leftmap and word[i] in rightmap and \
              leftmap[pattern[i]]==word[i] and rightmap[word[i]]==pattern[i]:
                  continue
              
              return False
  
          return True
  
      def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
          ans = []
          for word in words:
              if self.ispattern(word, pattern):
                  ans.append(word)
  
          return ans
  ```

* 模式编码

  ```python
  class Solution:
  
      def pattern_code(self, word):
          hashmap = {}
          i = 0
          for w in word:
              if w in hashmap:
                  continue
              hashmap[w] = i
              i += 1
          
          code = [hashmap[w] for w in word]
          return code
          
  
      def findAndReplacePattern(self, words: List[str], pattern: str) -> List[str]:
          ans = []
          code = self.pattern_code(pattern)
          for word in words:
              if self.pattern_code(word)==code:
                  ans.append(word)
  
          return ans
  ```

### 115. 根据字符出现的频率排序

* 计数法

  ```python
  class Solution:
      def frequencySort(self, s: str) -> str:
  
          counts = {}
          for x in s:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          kv = list(zip(counts.values(), counts.keys()))
          kv = sorted(kv, key=lambda x:x[0], reverse=True)
  
          ans = []
          for v, k in kv:
              for _ in range(v):
                  ans.append(k)
  
          return ''.join(ans)
  ```

### 116. 四数相加II

* 分组 + hash

  ```python
  class Solution:
      def fourSumCount(self, A: List[int], B: List[int], C: List[int], D: List[int]) -> int:
  
          ab = {}
          for a in A:
              for b in B:
                  if a+b in ab:
                      ab[a+b] += 1
                  else:
                      ab[a+b] = 1
  
          cd = {}
          for c in C:
              for d in D:
                  if c+d in cd:
                      cd[c+d] += 1
                  else:
                      cd[c+d] = 1
  
          ans = 0
          for k, v in ab.items():
              if -k in cd:
                  ans += v*cd[-k]
  
          return ans
  ```

  

### 117. 大小为 K 且平均值大于等于阈值的子数组数目

* 滑动窗口

  ```python
  class Solution:
  
      def numOfSubarrays(self, arr: List[int], k: int, threshold: int) -> int:
          
          count = 0
          sums = sum(arr[:k])
          val = k*threshold
          for i in range(k, len(arr)):
              if sums >= val:
                  count += 1
              sums += arr[i] - arr[i-k]
  
          if sums >= val:
                  count += 1
          return count
  ```

### 118. 冗余连接

* 并查集

  ```python
  class Solution:
  
      def find_root(self, x, parent):
          x_root = x
          while parent[x_root]!=-1:
              x_root = parent[x_root]
          return x_root
      
      def union(self, x, y, parent):
          x_root = self.find_root(x, parent)
          y_root = self.find_root(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
  
  
      def findRedundantConnection(self, edges: List[List[int]]) -> List[int]:
          parent = [-1 for _ in range(len(edges)+1)]
          children = []
          for x, y in edges:
              if self.union(x, y, parent)==False:
                  return [x, y]
                  break
  
          return []
  ```

  

### 119. 等式方程可满足性

* 并查集

  ```python
  class Solution:
  
      def find_root(self, x, parent):
          x_root = x
          while x_root in parent:
              x_root = parent[x_root]
  
          return x_root
  
      def union(self, x, y, parent):
          x_root = self.find_root(x, parent)
          y_root = self.find_root(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
  
      def equationsPossible(self, equations: List[str]) -> bool:
          parent = {}
  
          for eq in equations:
              if eq[1]=='=':
                  self.union(eq[0], eq[-1], parent)
  
          for eq in equations:
              if eq[1]=='!':
                  x_root = self.find_root(eq[0], parent)
                  y_root = self.find_root(eq[-1], parent)
                  if x_root==y_root:
                      return False
  
          return True
  ```


### 120. 账户合并

* 并查集

  ```python
  class Solution:
  
      def find_root(self, x, parent):
          x_root = x
          while x_root in parent:
              x_root = parent[x_root]
          
          return x_root
  
      def union(self, x, y, parent):
          x_root = self.find_root(x, parent)
          y_root = self.find_root(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
  
  
      def accountsMerge(self, accounts: List[List[str]]) -> List[List[str]]:
          parent = {}
  
          for i in range(len(accounts)):
              j = 1
              while j < len(accounts[i])-1:
                  self.union(accounts[i][j], accounts[i][j+1], parent)
                  j += 1
          
          ans = {}
          for i in range(len(accounts)):
              if len(accounts[i])==1:
                  continue
  
              x_root = self.find_root(accounts[i][1], parent)
              if x_root in ans:
                  ans[x_root].extend(accounts[i][1:])
              else:
                  ans[x_root] = accounts[i]
  
          res = []
          for k, v in ans.items():
              name, emails = [v[0]], v[1:]
              emails = list(set(emails))
              emails.sort()
              res.append(name + emails)
  
          return res
  
  ```

  

### 121. 由斜杠划分区域

* 深度优先搜索

  ```python
  class Solution:
  
  
      def dfs(self, area, x, y):
          if x >= 0 and x < len(area) and y >= 0 and y < len(area[0]) \
          and area[x][y]==0:
              area[x][y] = 1
              self.dfs(area, x+1, y)
              self.dfs(area, x, y + 1)
              self.dfs(area, x-1, y)
              self.dfs(area, x, y-1)
  
  
      def regionsBySlashes(self, grid: List[str]) -> int:
          area = [[0 for _ in range(len(grid)*3)] for _ in range(len(grid)*3)]
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  x, y = 3*i+1, 3*j+1
                  if grid[i][j]=='/':
                      area[x][y] = 1
                      area[x+1][y-1] = 1
                      area[x-1][y+1] = 1
                  elif grid[i][j]=='\\':
                      area[x][y] = 1
                      area[x+1][y+1] = 1
                      area[x-1][y-1] = 1
  
          count = 0
          for i in range(len(area)):
              for j in range(len(area)):
                  if area[i][j]==0:
                      self.dfs(area, i, j)
                      count += 1
          
          return count
  
  ```

* 并查集

  ```python
  class Solution:
  
      def find(self, x, parent):
          x_root = x
          while parent[x_root]!=-1:
              x_root = parent[x_root]
          return x_root
  
      def union(self, x, y, parent):
          x_root = self.find(x, parent)
          y_root = self.find(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
  
      def regionsBySlashes(self, grid: List[str]) -> int:
          n = len(grid)
          parent = [-1 for _ in range(4*n*n)]
  
          for i in range(len(grid)):
              for j in range(len(grid[0])):
                  x = 4*(n*i + j)
                  if grid[i][j]=='/':
                      self.union(x, x+3, parent)
                      self.union(x+1, x+2, parent)
                  elif grid[i][j]=='\\':
                      self.union(x, x+1, parent)
                      self.union(x+2, x+3, parent)
                  else:
                      self.union(x, x+1, parent)
                      self.union(x+1, x+2, parent)
                      self.union(x+2, x+3, parent)
                  
                  if i > 0:
                      self.union(x, 4*(n*(i-1) + j) + 2, parent)
                  if j > 0:
                      self.union(x+3, 4*(n*i + j-1) + 1, parent)
  
          count = 0
          for i in range(len(parent)):
              if parent[i]==-1:
                  count +=1
                  
          return count
  ```

### 122. 移除最多的同行或同列石头

* 并查集

  ```python
  class Solution:
  
      def find(self, x, parent):
          x_root = x
          while parent[x_root]!=-1:
              x_root = parent[x_root]
          return x_root
  
      def union(self, x, y, parent):
          x_root = self.find(x, parent)
          y_root = self.find(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
      
      def removeStones(self, stones: List[List[int]]) -> int:
          rows = {}
          cols = {}
          parent = [-1 for _ in range(len(stones))]
          for i in range(len(stones)):
              r, c = stones[i]
              if r in rows:
                  self.union(i, rows[r], parent)
              else:
                  rows[r] = i
  
              if c in cols:
                  self.union(i, cols[c], parent)
              else:
                  cols[c] = i
  
          count = 0
          for i in range(len(parent)):
              if parent[i]==-1:
                  count += 1
  
          return len(stones)-count
          
  ```

  

### 123. 连通网络操作次数

* 并查集

  ```python
  class Solution:
  
      def find(self, x, parent):
          x_root = x
          while parent[x_root]!=-1:
              x_root = parent[x_root]
          return x_root
  
      def union(self, x, y, parent):
          x_root = self.find(x, parent)
          y_root = self.find(y, parent)
          if x_root==y_root:
              return False
          else:
              parent[x_root] = y_root
              return True
  
      def makeConnected(self, n: int, connections: List[List[int]]) -> int:
  
          parent = [-1 for _ in range(n)]
          count = 0
          for x, y in connections:
              if self.union(x, y, parent)==False:
                  count += 1
          
          c = 0
          for i in range(len(parent)):
              if parent[i]==-1:
                  c += 1
  
          return c-1 if count >= c-1 else -1
  ```

  

### 124. 设计循环队列

* 长度变量

  ```python
  class MyCircularQueue:
  
      def __init__(self, k: int):
          """
          Initialize your data structure here. Set the size of the queue to be k.
          """
          self.queue = [0 for _ in range(k)]
          self.left = 0
          self.right = 0
          self.length = 0
          
  
      def enQueue(self, value: int) -> bool:
          """
          Insert an element into the circular queue. Return true if the operation is successful.
          """
          if self.isFull():
              return False
          else:
              self.queue[self.right] = value
              self.right = (self.right + 1)%len(self.queue)
              self.length += 1
              return True
  
          
  
      def deQueue(self) -> bool:
          """
          Delete an element from the circular queue. Return true if the operation is successful.
          """
          if self.isEmpty():
              return False
          else:
              font = self.queue[self.left]
              self.left = (self.left + 1)%len(self.queue)
              self.length -= 1
              return True
  
  
          
  
      def Front(self) -> int:
          """
          Get the front item from the queue.
          """
          if self.isEmpty():
              return -1
          return self.queue[self.left]
          
  
      def Rear(self) -> int:
          """
          Get the last item from the queue.
          """
          if self.isEmpty():
              return -1
          return self.queue[(self.right + len(self.queue) - 1)%len(self.queue)]
          
  
      def isEmpty(self) -> bool:
          """
          Checks whether the circular queue is empty or not.
          """
          return self.length==0
          
  
      def isFull(self) -> bool:
          """
          Checks whether the circular queue is full or not.
          """
          return self.length==len(self.queue)
          
          
  
  
  ```


### 125. 交换字符串使得字符串相同

* 计数法

  ```python
  class Solution:
      def minimumSwap(self, s1: str, s2: str) -> int:
          if len(s1)!=len(s2):
              return -1
  
          count_xy = 0
          count_yx = 0
          for i in range(len(s1)):
              if s1[i]=='x' and s2[i]=='y':
                  count_xy += 1
              elif s1[i]=='y' and s2[i]=='x':
                  count_yx += 1
  
          if count_xy%2==0 and count_yx%2==0:
              return (count_xy + count_yx)//2
          if count_xy%2==1 and count_yx%2==1:
              return (count_xy + count_yx)//2 + 1
  
          return -1
  ```

  

### 126. 数组大小减半

* 贪心算法

  ```python
  class Solution:
      def minSetSize(self, arr: List[int]) -> int:
  
          counts = {}
          for x in arr:
              if x in counts:
                  counts[x] += 1
              else:
                  counts[x] = 1
  
          vals = sorted(counts.values(), reverse=True)
          
          sums = 0
          sum_all = sum(vals)
          for i in range(len(vals)):
              sums += vals[i]
              if 2*sums >= sum_all:
                  return i+1
  ```

  

### 127. 打破字符串

* 排序

  ```python
  class Solution:
      def checkIfCanBreak(self, s1: str, s2: str) -> bool:
          if len(s1)!=len(s2):
              return False
          s1 = sorted(list(s1))
          s2 = sorted(list(s2))
          flag1 = True
          flag2 = True
          for i in range(len(s1)):
              if flag1==True and s1[i] < s2[i]:
                  flag1 = False
  
              if flag2==True and s1[i] > s2[i]:
                  flag2 = False
  
              if flag1==False and flag2==False:
                  return False
                  
          return True
  ```


### 128. 循环双端队列

* font/tail/length

  ```python
  class MyCircularDeque:
  
      def __init__(self, k: int):
          """
          Initialize your data structure here. Set the size of the deque to be k.
          """
          self.q = [None for _ in range(k)]
          self.length = 0
          self.font = 0
          self.tail = 0
          
  
      def insertFront(self, value: int) -> bool:
          """
          Adds an item at the front of Deque. Return true if the operation is successful.
          """
          if self.length==len(self.q):
              return False
  
          self.font = (self.font + len(self.q) - 1) % len(self.q)
          self.q[self.font] = value
          self.length += 1
  
          return True
          
  
      def insertLast(self, value: int) -> bool:
          """
          Adds an item at the rear of Deque. Return true if the operation is successful.
          """
          if self.length==len(self.q):
              return False
  
          self.q[self.tail] = value
          self.tail = (self.tail + 1) % len(self.q)
          self.length += 1
  
          return True
          
          
  
      def deleteFront(self) -> bool:
          """
          Deletes an item from the front of Deque. Return true if the operation is successful.
          """
          if self.length==0:
              return False
  
          self.font = (self.font + 1) % len(self.q)
          self.length -= 1
  
          return True
          
  
      def deleteLast(self) -> bool:
          """
          Deletes an item from the rear of Deque. Return true if the operation is successful.
          """
          if self.length==0:
              return False
  
          self.tail = (self.tail + len(self.q) - 1) % len(self.q)
          self.length -= 1
  
          return True
          
  
      def getFront(self) -> int:
          """
          Get the front item from the deque.
          """
          if self.length==0:
              return -1
  
          return self.q[self.font]
          
  
      def getRear(self) -> int:
          """
          Get the last item from the deque.
          """
          if self.length==0:
              return -1
              
          return self.q[(self.tail + len(self.q)-1) % len(self.q)]
          
  
      def isEmpty(self) -> bool:
          """
          Checks whether the circular deque is empty or not.
          """
          return self.length==0
          
  
      def isFull(self) -> bool:
          """
          Checks whether the circular deque is full or not.
          """
          return self.length==len(self.q)
          
  ```


### 129. 第K个数

* 动态规划

  ```python
  class Solution:
      def getKthMagicNumber(self, k: int) -> int:
  
          dp = [0 for _ in range(k)]
  
          p3 = 0
          p5 = 0
          p7 = 0
          dp[0] = 1
  
          for i in range(1, k):
              dp[i] = min(dp[p3]*3, dp[p5]*5, dp[p7]*7)
  
              if dp[i]==dp[p3]*3:
                  p3 += 1
              if dp[i]==dp[p5]*5:
                  p5 += 1
              if dp[i]==dp[p7]*7:
                  p7 += 1
  
          return dp[-1]
  ```

### 130. 区域和检索

* 线段树

  ```python
  class NumArray:
  
      def max_len(self, k):
          if k <= 2:
              return 2**k-1
  
          dp = [0 for _ in range(k+1)]
          dp[1] = 1
          dp[2] = 2
          for i in range(3, k+1):
              left = i//2
              right = i - left
              dp[i] = 1 + max(dp[left], dp[right])
  
          return 2**dp[-1] - 1
  
      def build_tree(self, node, left, right):
          if left==right:
              self.tree[node] = self.nums[left]
              return
  
          left_node = 2 * node + 1
          right_node = 2 * node + 2
  
          mid = (left + right)//2
          self.build_tree(left_node, left, mid)
          self.build_tree(right_node, mid + 1, right)
          self.tree[node] = self.tree[left_node] + self.tree[right_node]
  
      def update_tree(self, node, i, val, left, right):
          if left==right:
              self.tree[node] = val
              return
  
          left_node = 2 * node + 1
          right_node = 2 * node + 2
          
          mid = (left + right) // 2
          if i <= mid:
              self.update_tree(left_node, i, val, left, mid)
          else:
              self.update_tree(right_node, i, val, mid+1, right)
          
          self.tree[node] = self.tree[left_node] + self.tree[right_node]
  
      def __init__(self, nums: List[int]):
          if len(nums) > 0:
              self.nums = nums
              length = self.max_len(len(self.nums))
  
              self.tree = [0 for _ in range(length)]
              self.build_tree(0, 0, len(self.nums)-1)
  
          
  
      def update(self, i: int, val: int) -> None:
          self.update_tree(0, i, val, 0, len(self.nums)-1)
          self.nums[i] = val
  
  
      def query(self, node, i, j, left, right):
          if j < left or i > right:
              return 0
          elif i <= left and j >= right:
              return self.tree[node]
          else:
              left_node = 2 * node + 1
              right_node = 2 * node + 2
  
              mid = (left + right) // 2
              left_val = self.query(left_node, i, j, left, mid)
              right_val = self.query(right_node, i, j, mid+1, right)
              return left_val + right_val
          
  
  
  
      def sumRange(self, i: int, j: int) -> int:
          return self.query(0, i, j, 0, len(self.nums)-1)
      
  ```

  

### 132. 前K个高频单词

* 堆排序

  ```python
  class Solution:
  
  
      def build_heap(self, heap, n):
          i = (n-1)//2
          while i >= 0:
              self.heapify(heap, i, n)
              i -= 1
  
  
      def heapify(self, heap, i, n):
          left = 2 * i + 1 
          right = 2 * i + 2
          max_i = i
          if left < n:
              if heap[max_i][0] < heap[left][0]:
                  max_i = left
              elif heap[max_i][0] == heap[left][0]:
                  if heap[max_i][-1] > heap[left][-1]:
                      max_i = left
          if right < n:
              if heap[max_i][0] < heap[right][0]:
                  max_i = right
              elif heap[max_i][0] == heap[right][0]:
                  if heap[max_i][-1] > heap[right][-1]:
                      max_i = right
  
          if max_i==i:
              return
          else:
              temp = heap[max_i]
              heap[max_i] = heap[i]
              heap[i] = temp
              self.heapify(heap, max_i, n)
  
  
      def topKFrequent(self, words: List[str], k: int) -> List[str]:
  
          counts = {}
          for w in words:
              if w in counts:
                  counts[w] += 1
              else:
                  counts[w] = 1
  
          
          heap = [(freq, word) for word, freq in counts.items()]
          self.build_heap(heap, len(heap))
  
          ans = []
          end = len(heap)-1
          for _ in range(k):
              ans.append(heap[0][-1])
              temp = heap[0]
              heap[0] = heap[end]
              heap[end] = temp
              
              self.heapify(heap, 0, end)
              end -= 1
  
          return ans
  
  
  
  ```

* python内置堆

  ```python
  from collections import defaultdict
  from heapq import *
  class Solution:
  
      def topKFrequent(self, words: List[str], k: int) -> List[str]:
  
          counts = defaultdict(int)
          for w in words:
              counts[w] += 1
  
          heap = [(-freq, word) for word, freq in counts.items()]
  
          heapq.heapify(heap)
          
          return [heapq.heappop(heap)[-1] for _ in range(k)]
  ```

  

### 133. 数组中第k个最大的元素

* python内置堆

  ```python
  from heapq import *
  class Solution:
      def findKthLargest(self, nums: List[int], k: int) -> int:
  
          heapq.heapify(nums)
          return heapq.nlargest(k, nums)[-1]
  ```

* 堆排序

  ```python
  from heapq import *
  class Solution:
  
      def build_heap(self, nums, n):
          i = n//2 - 1
          while i >= 0:
              self.heapify(nums, i, n)
              i -= 1
  
      def heapify(self, nums, i, n):
          left = 2 * i + 1
          right = 2 * i + 2
          maxi = i
  
          if left < n and nums[maxi] < nums[left]:
              maxi = left
          if right < n and nums[maxi] < nums[right]:
              maxi = right
  
          if maxi!=i:
              temp = nums[maxi]
              nums[maxi] = nums[i]
              nums[i] = temp
              self.heapify(nums, maxi, n)
  
      def findKthLargest(self, nums: List[int], k: int) -> int:
  
          self.build_heap(nums, len(nums))
  
          end = len(nums)-1
          for _ in range(k):
              ans = nums[0]
              nums[0] = nums[end]
              nums[end] = ans
  
              self.heapify(nums, 0, end)
              end -= 1
  
          return ans
  ```

### 134. 最接近原点的K个点

* python内置堆

  ```python
  from heapq import *
  class Solution:
      def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
  
          heap = [(x**2 + y**2, [x, y]) for x, y in points]
  
          heapq.heapify(heap)
  
          return [heapq.heappop(heap)[-1] for _ in range(K)]
  ```

* 堆

  ```python
  from heapq import *
  class Solution:
  
      def build_heap(self, heap, n):
          i = n//2 - 1
          while i >= 0:
              self.heapify(heap, i, n)
              i -= 1
  
  
      def heapify(self, heap, i, n):
          left = 2 * i + 1
          right=  2 * i + 2
          mini = i
          if left < n and heap[mini][0] > heap[left][0]:
              mini = left
  
          if right < n and heap[mini][0] > heap[right][0]:
              mini = right
  
          if mini!=i:
              temp = heap[i]
              heap[i] = heap[mini]
              heap[mini] = temp
              self.heapify(heap, mini, n)
  
  
      def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
  
          heap = [(x**2 + y**2, [x, y]) for x, y in points]
  
          self.build_heap(heap, len(heap))
  
          ans = []
          end = len(heap)-1
          for _ in range(K):
              temp = heap[0]
              heap[0] = heap[end]
              heap[end] = temp
              ans.append(heap[end][-1])
  
              self.heapify(heap, 0, end)
              end -= 1
  
          return ans
  ```

  

### 135. 超级丑数

* 动态规划

  ```python
  class Solution:
      def nthSuperUglyNumber(self, n: int, primes: List[int]) -> int:
  
          primes.sort()
          index = [0 for _ in range(len(primes))]
  
          dp = [0 for _ in range(n)]
          dp[0] = 1
          for i in range(1, n):
              dp[i] = min([dp[index[j]]*primes[j] for j in range(len(index))])
              for j in range(len(index)):
                  if dp[i]==dp[index[j]]*primes[j]:
                      index[j] += 1
  
          return dp[-1]
  ```

### 136. 最小K个数

* 堆排序

  ```python
  class Solution:
  
      def swap(self, heap, i, j):
          temp = heap[i]
          heap[i] = heap[j]
          heap[j] = temp
  
      def build_heap(self, heap, n):
          i = n//2 - 1
          while i >= 0:
              self.heapify(heap, i, n)
              i -= 1
  
      def heapify(self, heap, i, n):
          left = 2*i + 1
          right = 2*i + 2
          mini = i
          if left < n and heap[left] < heap[mini]:
              mini = left
          if right < n and heap[right] < heap[mini]:
              mini = right
  
          if mini!=i:
              self.swap(heap, mini, i)
              self.heapify(heap, mini, n)
  
      def smallestK(self, arr: List[int], k: int) -> List[int]:
  
  
          self.build_heap(arr, len(arr))
  
          ans = []
          end = len(arr)-1
          for _ in range(k):
              self.swap(arr, 0, end)
              ans.append(arr[end])
  
              self.heapify(arr, 0, end)
              end -= 1
  
          return ans
  ```

* 快速排序

  ```python
  class Solution:
  
      
      def adjust(self, nums, low, high):
          key = nums[low]
          while low < high:
  
              while low < high and nums[high] >= key:
                  high -= 1
              
              if low < high:
                  nums[low] = nums[high]
              
              while low < high and nums[low] <= key:
                  low += 1
  
              if low < high:
                  nums[high] = nums[low]
  
          nums[low] = key
          return low
  
      def quick_sort(self, nums, low, high):
          if low < high:
              i = self.adjust(nums, low, high)
              self.quick_sort(nums, low, i-1)
              self.quick_sort(nums, i+1, high)
  
      def smallestK(self, arr: List[int], k: int) -> List[int]:
  
          self.quick_sort(arr, 0, len(arr)-1)
          return arr[:k]
  ```

* python内置堆

  ```python
  import heapq
  class Solution:
  
      def smallestK(self, arr: List[int], k: int) -> List[int]:
  
          heapq.heapify(arr)
          return [heapq.heappop(arr) for _ in range(k)]
  ```

  