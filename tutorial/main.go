package main

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"sync"
	"time"
)

type gasEngine struct {
	mpg       uint8
	gallons   uint8
	ownerInfo owner
}

type owner struct {
	name string
}

func (e gasEngine) milesLeft() uint8 {
	return e.gallons * e.mpg
}

var m = sync.RWMutex{} // RWMutex (can also use sync.Mutex)
var wg = sync.WaitGroup{}
var dbData = []string{"id1", "id2", "id3", "id4", "id5"}
var results = []string{}

func solutionWithMutex() {
	fmt.Println("=== THE SOLUTION: MUTEX LOCK/UNLOCK ===")

	t0 := time.Now()
	for i := 0; i < len(dbData); i++ {
		wg.Add(1)
		go dbCall(i)
	}
	wg.Wait()

	fmt.Printf("Total execution time: %v\n", time.Since(t0))
	fmt.Printf("The results are: %v\n", results)
	fmt.Printf("Successfully got %d results (no race condition!)\n", len(results))
	fmt.Println()
}

func dbCall(i int) {
	defer wg.Done()

	// Simulate DB delay
	var delay float32 = 2000 // Fixed delay for predictable output
	time.Sleep(time.Duration(delay) * time.Millisecond)

	fmt.Printf("The result from the database is: %s\n", dbData[i])

	// CRITICAL SECTION: Only one goroutine can execute this at a time
	m.Lock()                             // 🔒 LOCK: "I'm using the shared resource"
	results = append(results, dbData[i]) // Safe modification
	m.Unlock()                           // 🔓 UNLOCK: "I'm done, others can use it"
}

// ====================
// 3. HOW MUTEX WORKS INTERNALLY
// ====================

func howMutexWorks() {
	fmt.Println("=== HOW MUTEX WORKS INTERNALLY ===")

	var mutex sync.Mutex
	var counter int
	var wg2 sync.WaitGroup

	// Start 5 goroutines that increment a counter
	for i := 1; i <= 5; i++ {
		wg2.Add(1)
		go func(id int) {
			defer wg2.Done()

			fmt.Printf("Goroutine %d: Trying to acquire lock...\n", id)

			mutex.Lock() // 🔒 Only ONE goroutine can pass this line at a time
			fmt.Printf("Goroutine %d: 🔒 Got the lock! Current counter: %d\n", id, counter)

			// Critical section - only one goroutine executes this at a time
			oldValue := counter
			time.Sleep(100 * time.Millisecond) // Simulate work
			counter = oldValue + 1

			fmt.Printf("Goroutine %d: Incremented counter to %d\n", id, counter)
			fmt.Printf("Goroutine %d: 🔓 Releasing lock...\n", id)

			mutex.Unlock() // 🔓 Other goroutines can now acquire the lock

		}(i)
	}

	wg2.Wait()
	fmt.Printf("Final counter value: %d (should be 5)\n", counter)
	fmt.Println()
}

var MAX_CHICKEN_PRICE float64 = 5

func checkChickenPrices(website string, chickenChannel chan string) {
	for {
		time.Sleep(time.Second + 1)
		var chickenPrice = rand.Float32() * 20
		if chickenPrice <= float32(MAX_CHICKEN_PRICE) {
			chickenChannel <- website
			break
		}
	}
}

func main() {

	// testing channels
	var chickenChannel = make(chan string)
	var tofuChannel = make(chan string)
	var websites = []string{"google", "walmart"}
	for i := range websites {
		go checkChickenPrices(websites[i], chickenChannel)
	}

	var newChannel = make(chan int) // this channel can only a single syntax value
	newChannel <- 1                 // adding one to a channel
	var extractedChannelVal = <-newChannel

	var myEngine gasEngine = gasEngine{25, 15, owner{"Rumon"}}

	// fmt.Println("Hello World")
	array()

	// var intNum int = 32676

	// fmt.Println((intNum))

	// var myString string = "Hello" + "" + "World"
	// fmt.Println(myString)

	// fmt.Println(len("test"))

	// fmt.Println(utf8.RuneCountInString("Y"))

	// var my bool = false

	// myVar := "string"
	var printValue string = "Hellow World"
	var num int = 11
	var denominator int = 2
	var res, remainder, err = intDivision(num, denominator)
	if err != nil {
		fmt.Println((err.Error()))
	} else if remainder == 0 {
		fmt.Println("The result is", res)
	}
	fmt.Printf("The result of the int division is %v with remainder %v", res, remainder)
	printMe(printValue)

}

func printMe(printValue string) {
	fmt.Println(printValue)
}

func intDivision(numerator int, denominator int) (int, int, error) {
	var err error
	if denominator == 0 {
		err = errors.New("cannot divide by zero")
		return 0, 0, err
	}
	var result int = numerator / denominator
	var remainder int = numerator % denominator
	return result, remainder, err
}

func array() {
	var intArr []int32 = []int32{1, 3, 444} // not adding length value turns it into a slice
	intArr = append(intArr, 7)

	var intNewSlice []int32 = []int32{3, 4, 4}
	intNewSlice = append(intNewSlice, intArr...)
	fmt.Println(intNewSlice)

	// var intSlice3 []int32 = make([]int32, 3, 8)

	// // maps
	// var myMap map[string]uint8 = make(map[string]uint8)

	var myMap2 = map[string]uint8{"Adam": 23, "Sarah": 43}

	var age, ok = myMap2["Adam"]

	if ok {
		fmt.Printf("The age is %v\n", age)
	} else {
		fmt.Println("Invalid key")
	}

	for name := range myMap2 {
		fmt.Printf("%v\n", name)
	}

	// for i, v := range intArr {

	// }

	// while loop equivalent in go using for
	var i int = 0
	for i < 10 {
		fmt.Println((i))
		i = i + 1
	}

	// omitting the condition and using break
	for {
		if i >= 10 {
			break
		}
		fmt.Println(i)
		i = i + 1
	}

	// traditional
	var indexCheck int = 100
	for i := range indexCheck {
		fmt.Println(i)
	}
}

// an array with make preset capacity will take shorter time to append elements

func timeLoop(slice []int, n int) time.Duration {
	var t0 = time.Now()
	for len(slice) < n {
		slice = append(slice, 1)
	}
	return time.Since(t0)
}

func stringCheck() {
	var myString = "resume"
	var indexed = myString[0]
	fmt.Printf("%v, %T\n", indexed, indexed)
	for i, v := range myString {
		fmt.Println(i, v)
	}
}

// getting the max and min rows in order to calculate the area of hte rectangle
func minimumArea(grid [][]int) int {
	var maxRow int = 0
	var minRow int = math.MaxInt32
	var maxCol int = 0
	var minCol int = math.MaxInt32

	for row := 0; row < len(grid); row++ {
		for col := 0; col < len(grid[row]); col++ {
			currCell := grid[row][col]
			if currCell == 1 {
				maxRow = max(maxRow, row)
				minRow = min(minRow, row)
				maxCol = max(maxCol, col)
				minCol = min(minCol, col)
			}
		}
	}

	height := maxRow - minRow + 1
	width := maxCol - minCol + 1

	var area int = height * width
	return area
}

// mininum deletion
func minimumDeleteSum(s1 string, s2 string) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(indexOne, indexTwo int) int {
		cacheKey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)
		if value, found := memo[cacheKey]; found {
			return value
		}
		if indexOne == len(s1) && indexTwo == len(s2) {
			return 0
		}
		if indexOne == len(s1) {
			remainingSum := 0
			for index := indexTwo; index < len(s2); index++ {
				remainingSum = remainingSum + int(s2[index])
			}
			return remainingSum
		}
		if indexTwo == len(s2) {
			remainingSum := 0
			for index := indexOne; index < len(s1); index++ {
				remainingSum = remainingSum + int(s1[index])
			}
			return remainingSum
		}
		minSum := math.MaxInt32
		if s1[indexOne] == s2[indexTwo] {
			minSum = recurse(indexOne+1, indexTwo+1)
		} else {
			deleteFromS1 := int(s1[indexOne]) + recurse(indexOne+1, indexTwo)
			deleteFromS2 := int(s2[indexTwo]) + recurse(indexOne, indexTwo+1)
			minSum = min(deleteFromS1, deleteFromS2)
		}

		memo[cacheKey] = minSum
		return minSum
	}
	return recurse(0, 0)
}

func findTargetSumWays(nums []int, target int) int {
	memo := make(map[[2]int]int)

	var dfs func(int, int) int // declaring a variable to store the dfs go function

	dfs = func(index, currSum int) int {
		var cacheKey = [2]int{index, currSum}
		// getting cached values
		if cachedVal, found := memo[cacheKey]; found {
			return cachedVal
		}
		// base case
		if index >= len(nums) {
			if target == currSum {
				return 1
			}
			return 0
		}

		totalWays := dfs(index+1, nums[index]+currSum) + dfs(index+1, currSum-nums[index])

		memo[cacheKey] = totalWays
		return totalWays
	}

	return dfs(0, 0)
}

func isRob(nums []int) int {
	memo := make(map[int]int)

	var recurse func(int) int

	recurse = func(currIndex int) int {
		var cacheKey int = currIndex

		if keyValue, found := memo[cacheKey]; found {
			return keyValue
		}

		if currIndex >= len(nums) {
			return 0
		}

		// skipping a house then adding the next one
		currentHouse := nums[currIndex]
		includeCurrent := currentHouse + recurse(currIndex+2)
		skipCurrent := recurse(currIndex + 1)

		totalCurrentVal := 0
		totalCurrentVal = max(includeCurrent, skipCurrent)

		memo[cacheKey] = totalCurrentVal
		return totalCurrentVal
	}

	return recurse(0)
}

// wild card matching in rust
func isMatch(s string, p string) bool {
	memo := make(map[string]bool)

	var dfs func(int, int) bool

	dfs = func(indexOne, indexTwo int) bool {
		cacheKey := strconv.Itoa(indexOne) + "-" + strconv.Itoa(indexTwo)

		if value, found := memo[cacheKey]; found {
			return value
		}

		if indexTwo >= len(p) {
			return indexOne >= len(s)
		}

		if indexOne >= len(s) {
			// checking the rest of the string
			for index := indexTwo; index < len(p); index++ {
				if p[index] != '*' {
					return false
				}
			}
			return true
		}

		// if the characters are equal then do nothing
		var path bool = false
		if s[indexOne] == p[indexTwo] {
			path = dfs(indexOne+1, indexTwo+1)
		} else {
			if p[indexTwo] == '?' {
				path = dfs(indexOne+1, indexTwo+1)
			}
			if p[indexTwo] == '*' {
				path = dfs(indexOne+1, indexTwo) || dfs(indexOne, indexTwo+1)
			}
		}

		memo[cacheKey] = path
		return path

	}

	return dfs(0, 0)
}

// go version of dfs problem of restoring Ip address
// no need for caching as it will explore all possible string combinations

func restoreIdAddresses(s string) []string {

	ipCollection := []string{}
	dotLimit := 4

	isValidIp := func(currIp string) bool {
		isIpNum, _ := strconv.Atoi(currIp) // this method returns two terms
		return isIpNum < 256
	}

	var recurse func(int, int, string)

	recurse = func(currIndex, currDots int, currIp string) {

		if currIndex >= len(s) && currDots == dotLimit {
			ipAddress := currIp[:len(currIp)-1] // slicing it from the last char to remove the additional dots
			ipCollection = append(ipCollection, ipAddress)
			return
		}

		for index := currIndex; index < min(len(s), currIndex+3); index++ {
			currIpSlice := s[currIndex : index+1]

			if index != currIndex && s[currIndex] == '0' {
				continue
			}
			if isValidIp(currIpSlice) {
				// string concatenation only works with double quotes not runes
				recurse(index+1, currDots+1, currIp+currIpSlice+".")
			}

		}
	}
	recurse(0, 0, "")
	return ipCollection
}

// concatenating words
func findAllConcatenatedWordsInADict(words []string) []string {
	var mainSet = make(map[string]bool)
	var cache = make(map[string]bool)
	for _, word := range words {
		mainSet[word] = true
	}
	finalCollection := []string{}

	var recurse func(int, string, map[string]bool, int) bool

	recurse = func(currIndex int, currWord string, currSet map[string]bool, currWordCount int) bool {
		cacheKey := strconv.Itoa(currIndex) + "-" + currWord
		if val, found := cache[cacheKey]; found {
			return val
		}
		if currIndex > len(currWord) {
			cache[cacheKey] = false
			return false
		}
		// main base case
		if currIndex >= len(currWord) && currWordCount >= 2 {
			cache[cacheKey] = true
			return true
		}
		var validPath bool = false
		for index := currIndex + 1; index <= len(currWord); index++ {
			currSlice := currWord[currIndex:index]
			if mainSet[currSlice] {
				var foundPath bool = recurse(index, currWord, mainSet, currWordCount+1)
				validPath = foundPath || validPath
				if validPath {
					break
				}
			}
		}
		cache[cacheKey] = validPath
		return validPath
	}
	for _, word := range words {
		mainSet[word] = false
		// only add the if the word returns true after recursive structure succeeds
		if recurse(0, word, mainSet, 0) {
			finalCollection = append(finalCollection, word)
		}
		mainSet[word] = true
	}

	return finalCollection
}

// generating parenthesis
func generateParenthesis(n int) []string {
	result := []string{}
	var recurse func(int, int, string)
	// will return all paths
	recurse = func(openCount, closeCount int, substring string) {
		if openCount > n || closeCount > n {
			return
		}
		if openCount == n && closeCount == n {
			result = append(result, substring)
			return
		}
		if openCount < n {
			recurse(openCount+1, closeCount, substring+"(")
		}
		if closeCount < openCount {
			recurse(openCount, closeCount+1, substring+")")
		}
	}
	recurse(0, 0, "")
	return result

}

// hard problem
func cherryPick(grid [][]int) int {
	cache := make(map[string]int)
	rowLen := len(grid)
	var recurse func(int, int, int, int) int
	recurse = func(row1, col1, row2, col2 int) int {
		var cacheKey string = strconv.Itoa(row1) + "," + strconv.Itoa(col1) + "," + strconv.Itoa(row2) + "," + strconv.Itoa(col2)
		if value, found := cache[cacheKey]; found {
			return value
		}
		if row1 >= rowLen || row2 >= rowLen || col1 >= rowLen || col2 >= rowLen || grid[row1][col1] == -1 || grid[row2][col2] == -1 {
			return -1
		}
		if row1 == rowLen-1 && row2 == rowLen-1 && col1 == rowLen-1 && col2 == rowLen-1 {
			return grid[row1][col1]
		}
		maxCount := -1
		cherryCount := 0

		if row1 == row2 && col1 == col2 {
			cherryCount = grid[row1][col1]
		} else {
			cherryCount = grid[row1][col1] + grid[row2][col2]
		}
		maxCount = max(maxCount, recurse(row1+1, col1, row2+1, col2))
		maxCount = max(maxCount, recurse(row1+1, col1, row2, col2+1))
		maxCount = max(maxCount, recurse(row1, col1+1, row2+1, col2))
		maxCount = max(maxCount, recurse(row1, col1+1, row2, col2+1))

		if maxCount == -1 {
			cache[cacheKey] = -1
			return -1
		}

		totalCount := maxCount + cherryCount
		cache[cacheKey] = totalCount
		return totalCount
	}
	result := recurse(0, 0, 0, 0)
	if result == -1 {
		return 0
	} else {
		return result
	}
}

// getting the obstacle grid using a recursive approach using dfs memoization to record paths that have already been visited
func uniquePaths(obstacleGrid [][]int) int {
	cachePath := make(map[string]int)
	obstacle := 1
	rowLen := len(obstacleGrid)
	colLen := len(obstacleGrid[0])

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + strconv.Itoa(col)
		if value, found := cachePath[cacheKey]; found {
			return value
		}
		// base path
		if row >= rowLen || col >= colLen || row < 0 || col < 0 || obstacleGrid[row][col] == obstacle {
			return 0
		}

		if row == rowLen-1 && col == colLen-1 {
			return 1
		}

		totalPath := 0

		currPathTree := recurse(row+1, col) + recurse(row, col+1)
		totalPath = currPathTree + totalPath

		cachePath[cacheKey] = totalPath
		return totalPath

	}

	return recurse(0, 0)
}

// function to get the min Path cost for a matrix
func minPathCost(grid [][]int, moveCost [][]int) int {
	cache := make(map[string]int)
	rowLen := len(grid)
	colLen := len(grid[0])
	minCost := math.MaxInt32

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if value, found := cache[cacheKey]; found {
			return value
		}

		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return 0
		}
		// if its the last element then return the actual grid value
		if row == rowLen-1 {
			return grid[row][col]
		}
		minCost := math.MaxFloat32

		for nextCol := 0; nextCol < colLen; nextCol++ {
			currGridVal := grid[row][col]
			currMoveCost := moveCost[grid[row][col]][nextCol] // gets the exact index and col value of the movecost
			minCost = min(minCost, float64(currGridVal+currMoveCost+recurse(row+1, nextCol)))
		}
		cache[cacheKey] = int(minCost)
		return int(minCost)
	}

	for col := 0; col < colLen; col++ {
		minCost = min(recurse(0, col), minCost)
	}
	return minCost
}

// getting the min failing sum for path using recursiona and dfs approach TLE
func minFaillingSumHard(grid [][]int) int {
	cache := make(map[string]int)
	minCost := math.MaxFloat32
	rowLen := len(grid)
	colLen := len(grid[0])
	maxVal := 9999999

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if value, found := cache[cacheKey]; found {
			return value
		}
		// base border check
		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return maxVal
		}
		if row == rowLen-1 {
			return grid[row][col]
		}
		minCost := maxVal

		// traversing all the cols and checking whether the next one or not
		for nextCol := 0; nextCol < colLen; nextCol++ {
			if nextCol != col {
				minCost = min(minCost, recurse(row+1, nextCol))
			}
		}
		result := grid[row][col] + minCost
		cache[cacheKey] = result
		return result
	}

	for col := 0; col < colLen; col++ {
		minCost = min(minCost, float64(recurse(0, col)))
	}

	return int(minCost)
}

// getting the longest increasing path
func longestIncreasingPath(matrix [][]int) int {
	cache := make(map[string]int)
	longestPathCount := 0
	rowLen := len(matrix)
	colLen := len(matrix[0])

	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if cachedValue, found := cache[cacheKey]; found {
			return cachedValue
		}
		// base case - exceed boundary and return 0 since there is no path
		if row < 0 || col < 0 || row >= rowLen || col >= colLen {
			return 0
		}
		directions := [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}}
		localMaxPathResult := 0
		for _, direction := range directions {
			// need to extrapolate next row or next col
			newRow := row + direction[0]
			newCol := col + direction[1]
			// border check then only recurse if the condition passes for valid path
			if newRow >= 0 && newCol >= 0 && newRow < rowLen && newCol < colLen {
				if matrix[newRow][newCol] > matrix[row][col] {
					// updates the particular direction and each recursive cycle
					localMaxPathResult = max(localMaxPathResult, recurse(newRow, newCol))
				}
			}
		}
		// adds one if the localResult returns a path..
		localResult := 1 + localMaxPathResult
		cache[cacheKey] = localResult
		return localResult
	}

	// traversing the matrix and updating the current max path
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			currMaxPath := recurse(row, col)
			longestPathCount = max(longestPathCount, currMaxPath)
		}
	}

	return longestPathCount
}

// using dfs memo to calculate the count paths from a grid
func countPaths(grid [][]int) int {
	cache := make(map[string]int)
	paths := 0
	rowLen := len(grid)
	colLen := len(grid[0])
	const MOD = 1000000007 // to limit the mod path and control overflow
	// Optimized caching approach
	visited := make([][]int, len(grid))
	for index := 0; index < len(visited); index++ {
		visited[index] = make([]int, len(visited[index]))
		for subIndex := 0; subIndex < len(visited[index]); subIndex++ {
			visited[index][subIndex] = -1
		}
	}
	var recurse func(int, int) int
	recurse = func(row, col int) int {
		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if cachedValue, found := cache[cacheKey]; found {
			return cachedValue
		}
		if row < 0 || row >= rowLen || col < 0 || col >= colLen {
			return 0
		}
		var directions = [][]int{{0, 1}, {0, -1}, {1, 0}, {-1, 0}} // directions for the grid path to traverse in
		localPath := 1
		for _, direction := range directions {
			newRow := row + direction[0]
			newCol := col + direction[1]

			if newRow >= 0 && newRow < rowLen && newCol >= 0 && newCol < colLen && grid[newRow][newCol] > grid[row][col] {
				localPath = (localPath + recurse(newRow, newCol)) % MOD
			}
		}
		cache[cacheKey] = localPath
		return localPath
	}
	// passing each cell location in order to get the total number of paths
	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			paths = (paths + recurse(row, col)) % MOD
		}
	}
	return paths
}

// getting maxmimum gold with dfs memoization ... visiting a cell with 0 resets the path to return 0
func getMaximumGold(grid [][]int) int {
	maxAmount := 0
	rowLen := len(grid)
	colLen := len(grid[0])

	var recurse func(int, int, map[string]bool) int
	recurse = func(row, col int, visited_map map[string]bool) int {

		cacheKey := strconv.Itoa(row) + "," + strconv.Itoa(col)
		if visited_map[cacheKey] {
			return 0
		}

		visited_map[cacheKey] = true // visited path

		if row < 0 || row >= rowLen || col < 0 || col >= colLen || grid[row][col] == 0 {
			return 0
		}

		directions := [][]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}
		localMax := 0

		for _, direction := range directions {
			newRow := row + direction[0]
			newCol := col + direction[1]

			if newRow >= 0 && newCol >= 0 && newRow < rowLen && newCol < colLen &&
				grid[newRow][newCol] != 0 {
				localMax = max(localMax, recurse(newRow, newCol, visited_map))
			}
		}
		visited_map[cacheKey] = false // unmark for backtracking
		totalPath := grid[row][col] + localMax
		return totalPath
	}

	for row := 0; row < rowLen; row++ {
		for col := 0; col < colLen; col++ {
			if grid[row][col] != 0 {
				maxAmount = max(maxAmount, recurse(row, col, make(map[string]bool)))
			}
		}
	}

	return maxAmount
}

// difference is changing the grid in place
func optimizedPathGold(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	maxAmount := 0
	rows := len(grid)
	cols := len(grid[0])

	var backtrack func(int, int) int
	backtrack = func(row, col int) int {
		// Check bounds and validity
		if row < 0 || row >= rows || col < 0 || col >= cols || grid[row][col] == 0 {
			return 0
		}

		// Store original value and mark as visited
		original := grid[row][col]
		grid[row][col] = 0

		// Explore all 4 directions
		directions := [][]int{{0, 1}, {0, -1}, {-1, 0}, {1, 0}}
		maxPath := 0

		for _, dir := range directions {
			newRow := row + dir[0]
			newCol := col + dir[1]
			maxPath = max(maxPath, backtrack(newRow, newCol))
		}

		// Backtrack: restore original value
		grid[row][col] = original

		return original + maxPath
	}

	// Try starting from each cell with gold
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			if grid[row][col] != 0 {
				maxAmount = max(maxAmount, backtrack(row, col))
			}
		}
	}

	return maxAmount
}

// HARD Problem: getting number of steps to get to 0 using recursion
func numWays(steps int, arrLen int) int {
	numWays := 0
	const MOD = 1000000007
	// limit array length before creating memo
	if arrLen > steps+1 {
		arrLen = steps + 1
	}

	maxPos := min(steps, arrLen-1)
	memo := make([][]int, arrLen)
	for i := range memo {
		memo[i] = make([]int, steps+1)
		for j := range memo[i] {
			memo[i][j] = -1
		}
	}
	var recurse func(int, int) int
	recurse = func(currentPos, remainingSteps int) int {
		// returning a cached path
		if memo[currentPos][remainingSteps] != -1 {
			return memo[currentPos][remainingSteps]
		}
		// main base case to return a valid path
		if remainingSteps == 0 {
			// main case to return a valid path
			if currentPos == 0 {
				return 1
			}
			return 0
		}
		// if steps are exchausted then there is no turning back to 0
		if currentPos > remainingSteps || remainingSteps < 0 || currentPos > maxPos {
			memo[currentPos][remainingSteps] = 0
			return 0
		}
		totalPaths := 0
		// moving right within boundary
		movingRight := 0
		if currentPos+1 < arrLen {
			movingRight = (recurse(currentPos+1, remainingSteps-1)) % MOD
		}
		// moving left within boundary
		movingLeft := 0
		if currentPos-1 >= 0 {
			movingLeft = (recurse(currentPos-1, remainingSteps-1)) % MOD
		}
		samePos := 0
		samePos = (recurse(currentPos, remainingSteps-1)) % MOD

		totalPaths = (movingLeft + movingRight + samePos) % MOD
		memo[currentPos][remainingSteps] = totalPaths
		return totalPaths
	}
	numWays = recurse(0, steps) // initial recursive call to begin the recursion
	return numWays
}

// simialar question to reaching destination but above is hard problem
func numberOfWays(startPos int, endPos int, k int) int {
	cache := make(map[string]int)
	const MOD = 1000000007

	var recurse func(int, int) int
	recurse = func(currentPos, remainingSteps int) int {
		cacheKey := strconv.Itoa(currentPos) + "-" + strconv.Itoa(remainingSteps)
		if val, found := cache[cacheKey]; found {
			return val
		}
		if remainingSteps < 0 {
			return 0
		}

		if remainingSteps == 0 {
			if currentPos == endPos {
				return 1
			}
			return 0
		}
		// no more steps remaining
		if math.Abs(float64(currentPos)-float64(endPos)) > float64(remainingSteps) {
			return 0
		}

		totalWays := 0
		left := recurse(currentPos-1, remainingSteps-1)
		right := recurse(currentPos+1, remainingSteps-1)

		totalWays = (left + right) % MOD

		cache[cacheKey] = totalWays
		return totalWays
	}

	return recurse(startPos, k)
}

// minimum cost to cut stick... cutting from both sides using dfs memo interval technique
func minCostToCutStick(n int, cuts []int) int {
	cuts = append(cuts, 0, n)
	sort.Ints(cuts) // enables us to recurse between intervals
	minCost := math.MaxInt32

	var recurse func(int, int, map[string]int) int
	recurse = func(left, right int, cache map[string]int) int {
		// cached minimum path
		cacheKey := strconv.Itoa(left) + "-" + strconv.Itoa(right)
		if cacheMin, found := cache[cacheKey]; found {
			return cacheMin
		}
		// base case no possible cuts to check if there is any cut positions between the left and rigth boundary
		if right-left <= 1 {
			return 0
		}
		minCostLocal := math.MaxInt32

		// for loop to check through various intervals
		for currIndex := left + 1; currIndex < right; currIndex++ {
			currentCost := cuts[right] - cuts[left]
			minCost := currentCost + recurse(left, currIndex, cache) + recurse(currIndex, right, cache)
			minCostLocal = min(minCost, minCostLocal)
		}

		cache[cacheKey] = minCostLocal
		return minCostLocal
	}

	minCost = recurse(0, len(cuts)-1, make(map[string]int))
	// have to pass len cuts - 1 because its calculating till n value
	return minCost
}

// solving jump game V with dfs memo and recursion while keeping track of proper caching
func jumpGameV(arr []int, d int) int {
	maxCost := 0
	memo := make(map[int]int)
	// dfs function to check whether the inner elements are bigger than the current one or not
	var canJump func(int, int) bool
	canJump = func(from, to int) bool {
		// if the from side is smaller or equal then no more checks need to be performed
		if arr[from] <= arr[to] {
			return false
		}
		// checking elements within the from and to range
		start := min(from, to) + 1
		end := max(from, to)
		for i := start; i < end; i++ {
			currEl := arr[i]
			if currEl >= arr[from] {
				return false
			}
		}
		return true
	}
	// dfs recursive function
	var recurse func(int, int) int
	recurse = func(start, d int) int {
		// cached max cost
		cacheKey := start
		if cachedMaxCost, found := memo[cacheKey]; found {
			return cachedMaxCost
		}
		maxVisits := 1
		// checking left step jump count
		for step := 1; step <= d; step++ {
			newJumpIndex := start - step // goes left
			if newJumpIndex < 0 {        // bound checking
				break
			}
			if canJump(start, newJumpIndex) {
				visits := 1 + recurse(newJumpIndex, d)
				maxVisits = max(maxVisits, visits)
			}
		}
		// checking right step jump count
		for step := 1; step <= d; step++ {
			newJumpIndex := start + step  // for right
			if newJumpIndex >= len(arr) { // bound checking
				break
			}
			if canJump(start, newJumpIndex) {
				visits := 1 + recurse(newJumpIndex, d)
				maxVisits = max(maxVisits, visits)
			}
		}
		memo[cacheKey] = maxVisits
		return maxVisits
	}
	// starting from each position in the array and collect recursive paths
	for startIndex := 0; startIndex < len(arr); startIndex++ {
		result := recurse(startIndex, d)
		maxCost = max(maxCost, result)
	}
	return maxCost
}

// jump game ii reaching index with value 0
// record the visited index using a set

func jumpGameIII(arr []int, start int) bool {
	memo := make(map[int]bool)
	visitedIndex := make(map[int]bool)

	var recurse func(int) bool
	recurse = func(index int) bool {
		cacheKey := index
		if cachedPath, found := memo[cacheKey]; found {
			return cachedPath
		}
		visitedIndex[index] = true
		// main base case for true
		if arr[index] == 0 {
			return true
		}
		if index < 0 || index >= len(arr) {
			memo[cacheKey] = false
			return false
		}
		validPath := false
		currJumpValue := arr[index]
		// logic for left side
		var leftJump bool
		if index-currJumpValue >= 0 && !visitedIndex[index-currJumpValue] {
			leftJump = recurse(index - currJumpValue)
		}
		// logic for right side
		var rightJump bool
		if index+currJumpValue < len(arr) && !visitedIndex[index+currJumpValue] {
			rightJump = recurse(index + currJumpValue)
		}

		validPath = leftJump || rightJump
		memo[cacheKey] = validPath
		return validPath
	}

	state := recurse(start)
	return state
}

// frog can cross using dfs memo approach
func frogCanJump(stones []int) bool {
	// make boolean array
	maxPos := stones[len(stones)-1]
	memo := make(map[string]bool)

	if stones[1] != 1 {
		return false
	}

	hasStones := make(map[int]bool)
	for _, v := range stones {
		hasStones[v] = true
	}
	var recurse func(int, int) bool
	recurse = func(currIndex, prevJump int) bool {
		key := strconv.Itoa(currIndex) + "-" + strconv.Itoa(prevJump)
		if val, found := memo[key]; found {
			return val
		}
		// main base case
		if currIndex < 0 || currIndex > maxPos {
			memo[key] = false
			return false
		}
		if currIndex == stones[len(stones)-1] {
			return true
		}
		validJump := false
		// main check to see whether jump is valid or not
		directions := []int{prevJump - 1, prevJump, prevJump + 1}

		for _, direction := range directions {
			if direction > 0 {
				newDirection := direction + currIndex
				if hasStones[newDirection] { // checking stones
					jumpState := recurse(newDirection, direction) // direction is the previous direction
					validJump = jumpState || validJump
				}
			}
		}

		memo[key] = validJump
		return validJump
	}
	return recurse(1, 1) // passing current index and prevJump coords
}

// geting minimum side ways jump only jump if the obstacles position is not equal to lane
func minSideWaysJump(obstacles []int) int {
	cache := make(map[string]int)
	minJumps := math.MaxInt32

	var recurse func(int, int) int
	recurse = func(currPos, currLane int) int {
		// cached min jump path
		cacheKey := strconv.Itoa(currPos) + "-" + strconv.Itoa(currLane)
		if val, found := cache[cacheKey]; found {
			return val
		}
		// main base case if it exceeds length
		if currPos == len(obstacles)-1 {
			return 0
		}
		if currPos >= len(obstacles) {
			return 9999999
		}
		jumps := math.MaxInt32
		// keep the same path if there are no obstacles in the current lane
		if obstacles[currPos+1] != currLane && currPos+1 < len(obstacles) {
			jumps = min(recurse(currPos+1, currLane), jumps)
		}
		// checking for sideways jumps
		for lane := 1; lane <= 3; lane++ {
			// side jump plus moving forward in orer to avoid infinite recursion
			if lane != currLane && obstacles[currPos] != lane && obstacles[currPos+1] != lane {
				jumps = int(math.Min(float64(jumps), 1+float64(recurse(currPos+1, lane))))
			}
		}
		cache[cacheKey] = jumps
		return jumps
	}

	minJumps = recurse(0, 2)
	return minJumps
}

// using skip and reuse technique for dfs memoization
func combinationSum4(nums []int, target int) int {
	memo := make(map[string]int)

	var recurse func(int, int) int
	recurse = func(index, target int) int {
		key := strconv.Itoa(index) + "-" + strconv.Itoa(target)
		if val, found := memo[key]; found {
			return val
		}
		if target == 0 {
			return 1
		}
		if index >= len(nums) || target < 0 {
			return 0
		}
		// means u can pick again from any combination
		ways := recurse(0, target-nums[index]) // resets index to make it work for any number of other numbers

		skip := recurse(index+1, target)

		totalWays := ways + skip
		memo[key] = totalWays
		return totalWays
	}

	return recurse(0, target)
}

// getting the cheapest price for flights remember.. k flight means k stops means k + 1 stops
func findCheapestPrice(n int, flights [][]int, src int, dst int, k int) int {
	memo := make(map[string]int)
	minCost := math.MaxInt32
	// organizing array into map structure
	// array restructing in order to collect the items based on the source being the key and the to and cost being the key pair values
	flightMap := make(map[int][][2]int)
	for _, flight := range flights {
		from, to, cost := flight[0], flight[1], flight[2]
		flightMap[from] = append(flightMap[from], [2]int{to, cost})
	}
	// the recursion will be traversing through a map hence you will not need to use an index for traversal
	var recurse func(int, int) int
	recurse = func(city, remainingStops int) int {
		// cached minCost for flight traversal
		key := strconv.Itoa(city) + "-" + strconv.Itoa(remainingStops)
		if cachedJumpValue, found := memo[key]; found {
			return cachedJumpValue
		}
		// base case
		if city == dst {
			return 0
		}
		// if there are no remaining stops then return the maximum int size value possible
		if remainingStops < 0 {
			return math.MaxInt32
		}
		localMinCost := math.MaxInt32
		// current local cost from each source city
		for _, flightDetails := range flightMap[city] {
			destination, cost := flightDetails[0], flightDetails[1]
			if remainingStops > 0 { // need to check before running the recursion
				recursiveResult := recurse(destination, remainingStops-1)
				if recursiveResult != math.MaxInt32 {
					totalCost := recursiveResult + cost
					localMinCost = min(localMinCost, totalCost)
				}
			}
		}
		memo[key] = localMinCost
		return localMinCost
	}

	// getting the recursive avlue
	minCost = recurse(src, k+1)
	if minCost == math.MaxInt32 {
		return -1
	}
	return minCost
}
