from datetime import datetime, timedelta
import heapq
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from matplotlib.patches import Patch
import pandas as pd
import numpy as np

def ProcessBlock(lines, processed, filterOut):
	if len(lines[3].split()) < 2:
		print(lines)
		return
	if lines[6].split()[2] == "seconds":
		return
	title = lines[0]
	if title in filterOut:
		return
	players = int(lines[3].split()[1])
	specs = int(lines[4].split()[1])
	start = lines[5].split()[1] + ' ' + lines[5].split()[2] + ' ' + lines[5].split()[3]
	start = datetime.strptime(start, "%m/%d/%Y %I:%M:%S %p")
	duration = int(lines[6].split()[1])

	if title not in processed:
		processed[title] = []
	processed[title].append({
		"players" : players,
		"specs" : specs,
		"start" : start,
		"end" : start + timedelta(minutes=duration),
		"duration" : duration,
	})


def RollingAverage(data, windowSize):
	if windowSize % 2 == 0:
		raise ValueError("Window size must be odd for centering.")
	half_window = windowSize // 2
	smoothed = np.convolve(data, np.ones(windowSize)/windowSize, mode='valid')
	# Pad with NaNs (or extend the original times to match)
	padding = [np.nan] * half_window
	return padding + list(smoothed) + padding


def CountThresholdProp(timeline, threshold):
	count = 0
	for value in timeline:
		if value >= threshold:
			count += 1
	return count / len(timeline)


def SortBattles(a):
	return a["start"]


def MakeBattleList(processed):
	print('{},{},{},{},{}'.format("Room", "Players", "Time", "Duration", "Spectators"))
	for title, battles in processed.items():
		battles.sort(key=SortBattles)
		for data in battles:
			print('{},{},{},{},{}'.format(
				title, data["players"], data["start"].strftime("%Y-%m-%d %H:%M"),
				data["duration"], data["specs"]))


def PlotTimeline(times, dayAgg, data, minuteScale, weekAverage):
	data = data.copy()
	specs = data["specs"]
	del data["specs"]

	windowSize = 21
	specs = RollingAverage(specs, windowSize)

	labels = list(data.keys())[::-1]
	values = [data[label] for label in labels]

	fig, ax = plt.subplots(figsize=(12, 6))

	ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
	ax.grid(axis='y', linestyle='-', color='lightgray')
	ax.grid(axis='x', linestyle='-', color='lightgray')
	plt.axhline(y=32, linestyle='-', color='red', linewidth=0.5)
	plt.axhline(y=22, linestyle='-', color='red', linewidth=0.5)
	ax.set_axisbelow(True)
	print(list(dayAgg.values())[0]["end"])
	ax.set_xlim(list(dayAgg.values())[0]["end"] - timedelta(hours=24), list(dayAgg.values())[-1]["end"])

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(labels)]
	ax.stackplot(times, values, labels=labels, step='post', colors=colors[::-1])
	ax.plot(times, specs, label='Spectators', linestyle='--', color='black', linewidth=1)
	if weekAverage:
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	else:
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d/%m'))
	ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 6)))

	# Space for the labels at the top
	ymin, ymax = ax.get_ylim()
	ax.set_ylim(ymin, ymax + 7)

	# Add vertical lines and labels
	for dayData in dayAgg.values():
		duration = sum([count*i for i, count in enumerate(dayData["battleSizes"])]) / sum(dayData["battleSizes"])
		ax.axvline(x=dayData["end"], color='black', linestyle='--', linewidth=0.5)
		ax.text(dayData["start"] + timedelta(hours=2), ax.get_ylim()[1] - 1,
			"{}\n{:,.0f} pm\n{:,.0f} games\n{:,.1f} ave dur\n{:,.0f} with ≥ 16\n{:,.0f} with ≥ 22\n{:,.0f} with ≤ 10".format(
				(dayData["end"] - timedelta(hours=12)).strftime('%A'),
				dayData["playerminutes"],
				sum(dayData["battleSizes"]),
				duration,
				sum(dayData["battleSizes"][16:]),
				sum(dayData["battleSizes"][22:]),
				sum(dayData["battleSizes"][:10])),
			rotation=0, verticalalignment='top', horizontalalignment='left',
			backgroundcolor='white', fontsize=10)

	handles = [Patch(facecolor=colors[i], label=label) for i, label in enumerate(labels[::-1])]
	line_handle = Line2D([0], [0], color='black', linewidth=2, label='Spectators (smoothed)')
	handles.append(line_handle)
	ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.8), fontsize=7)
	if weekAverage:
		title = "Average players in team games"
	else:
		title = "Maximum players in team games"
	plt.title(
		'{} by {}-minute period {} to {}. Counts in playerminutes (pm). Game counts include ≥ 5 minutes and ≥ 4 players.'.format(
		title, minuteScale,
		times[0].strftime('%D %H:%M'),
		times[-1].strftime('%D %H:%M'),
	))
	plt.xlabel('Time')
	plt.ylabel('Players')

	plt.xticks(rotation=90)
	plt.tight_layout()
	plt.show()


def PrintTimeline(minTime, maxTime, minuteScale, playerLists):
	print("Time," + ",".join(playerLists.keys()))
	time = minTime
	index = 0
	while time < maxTime:
		str = time.strftime("%H:%M")
		for count in playerLists.values():
			str = str + ",{}".format(count[index])
		print(str)
		time = time + timedelta(minutes=minuteScale)
		index = index + 1


def GetDayEnd(time, offset):
	return (time - timedelta(hours=offset)).replace(hour=0, minute=0, second=0) + timedelta(hours=offset + 24)


def FlattenTimelines(counts, startIndex, index, mode):
	dayTimeline = False
	for key, data in counts.items():
		if key != "specs":
			if dayTimeline is False:
				dayTimeline = [0] * len(data[startIndex:index])
			if mode == 'sum':
				dayTimeline = [x + y for x, y in zip(dayTimeline, data[startIndex:index])]
			elif mode == 'max':
				dayTimeline = [max(x, y) for x, y in zip(dayTimeline, data[startIndex:index])]
	return dayTimeline


def GetDailyValues(times, maxData, averageData, step, offset, mostBattles, processed):
	averageData = averageData.copy()
	del averageData["specs"]

	outputs = {}
	nextDay = GetDayEnd(times[0], offset)
	lastTime = times[0]
	index = 0
	playerAcc = 0
	startIndex = 0
	gameAcc = 0
	for time in times:
		if time >= nextDay:
			outputs[nextDay] = {
				"start" : lastTime,
				"end" : nextDay, 
				"dayTimeline" : FlattenTimelines(maxData, startIndex, index, 'sum'),
				"dayMax" : FlattenTimelines(maxData, startIndex, index, 'max'),
				"playerminutes" : playerAcc*step,
				"battleSizes" : [0] * 40,
				"battleDurations" : [0] * 200,
			}
			nextDay = GetDayEnd(time, offset)
			lastTime = time
			playerAcc = 0
			startIndex = index
		for count in averageData.values():
			playerAcc += count[index]
		index += 1
	
	outputs[nextDay] = {
		"start" : lastTime,
		"end" : time, 
		"dayTimeline" : FlattenTimelines(maxData, startIndex, index, 'sum'),
		"dayMax" : FlattenTimelines(maxData, startIndex, index, 'max'),
		"playerminutes" : playerAcc*step,
		"battleSizes" : [0] * 40,
		"battleDurations" : [0] * 200,
	}

	for title in mostBattles:
		battles = processed[title]
		for data in battles:
			battleDay = GetDayEnd(data["start"], offset)
			if battleDay in outputs:
				if data["duration"] >= 5 and data["players"] >= 4:
					outputs[battleDay]["battleSizes"][data["players"]] += 1
					outputs[battleDay]["battleDurations"][data["duration"]] += 1
			else:
				print("Missing battle day {}".format(battleDay))

	return outputs


def MakeAverageCounts(processed, minTime, step, times, mostBattles):
	playerLists = {}
	totalSpecs = False
	for title in mostBattles:
		players = [0] * len(times)
		specs = [0] * len(times)
		battles = processed[title]
		for data in battles:
			index = math.floor((data["start"] - minTime) / step)
			time = minTime + index*step
			while time < data["end"] - step:
				if time < data["start"]:
					prop = (min(time + step, data["end"]) - data["start"]) / step
					players[index] += data["players"] * prop
					specs[index] += data["specs"] * prop
				else:
					players[index] += data["players"]
					specs[index] += data["specs"]
				time += step
				index += 1
			prop = (data["end"] - max(time, data["start"])) / step
			players[index] += data["players"] * prop
			specs[index] += data["specs"] * prop
		
		playerLists[title] = players
		if totalSpecs is False:
			totalSpecs = specs
		else:
			totalSpecs = [a + b for a, b in zip(specs, totalSpecs)]
	playerLists["specs"] = totalSpecs
	return playerLists


def MakeMaximumCounts(processed, minTime, step, times, mostBattles):
	playerLists = {}
	totalSpecs = False
	for title in mostBattles:
		players = [0] * len(times)
		specs = [0] * len(times)
		battles = processed[title]
		for data in battles:
			index = math.floor((data["start"] - minTime) / step)
			time = minTime + index*step
			while time < data["end"]:
				players[index] = max(players[index], data["players"])
				specs[index] = max(specs[index], data["specs"])
				time += step
				index += 1
		playerLists[title] = players
		if totalSpecs is False:
			totalSpecs = specs
		else:
			totalSpecs = [a + b for a, b in zip(specs, totalSpecs)]
	playerLists["specs"] = totalSpecs
	return playerLists


def GetWeekAverage(playerCounts, daily, minuteScale, minTime):
	weekCounts = {}
	weekSize = int(7*24*60 / minuteScale)
	for name, data in playerCounts.items():
		week = [0] * weekSize
		for x in range(weekSize):
			indices = range(x, len(data), weekSize)
			values = [data[i] for i in indices]
			week[x] = sum(values) / len(values)
		weekCounts[name] = week
	
	times = []
	time = minTime
	step = timedelta(minutes=minuteScale)
	while time < minTime + timedelta(days=7):
		times.append(time)
		time = time + step

	weekDaily = {}
	for day, data in daily.items():
		match = [x for x in weekDaily.keys() if x.weekday() == day.weekday()]
		if len(match) == 0:
			weekDaily[day] = data.copy()
			weekDaily[day]["copies"] = 1
			weekDaily[day]["playerminutesList"] = [weekDaily[day]["playerminutes"]]
			weekDaily[day]["dayTimelineList"] = [weekDaily[day]["dayTimeline"]]
			weekDaily[day]["dayMaxList"] = [weekDaily[day]["dayMax"]]
		else:
			match = match[0]
			weekDaily[match]["playerminutes"] += data["playerminutes"]
			weekDaily[match]["playerminutesList"].append(data["playerminutes"])
			weekDaily[match]["dayTimelineList"].append(data["dayTimeline"])
			weekDaily[match]["dayMaxList"].append(data["dayMax"])
			weekDaily[match]["battleSizes"] = [
				x + y for (x, y) in zip(weekDaily[match]["battleSizes"], data["battleSizes"])]
			weekDaily[match]["battleDurations"] = [
				x + y for (x, y) in zip(weekDaily[match]["battleDurations"], data["battleDurations"])]
			weekDaily[match]["copies"] += 1

	for day, data in weekDaily.items():
		data["playerminutes"] /= data["copies"]
		data["battleSizes"] = [x / data["copies"] for x in data["battleSizes"]]
		data["battleDurations"] = [x / data["copies"] for x in data["battleDurations"]]

	return weekCounts, weekDaily, times


def PrintWeekStats(weekDaily):
	for day, data in weekDaily.items():
		print("{}: mean {:,.0f}, min {:,.0f}, 33rd {:,.0f}, 67th {:,.0f}, max {:,.0f}".format(
			day.strftime('%A'),
			np.mean(data["playerminutesList"]),
			np.min(data["playerminutesList"]),
			np.percentile(data["playerminutesList"], 33),
			np.percentile(data["playerminutesList"], 67),
			np.max(data["playerminutesList"])))


def MakeBoxplot(xAxis, metricName, dayList, df, mainTimes, dfExtra, extraTimes, swarm):
	# Create the swarm plot
	plt.figure(figsize=(18, 9))
	ax = sns.boxplot(
		data=df,
		x=xAxis,
		y=metricName,
		order=dayList,
		width=0.3,
		showcaps=True,
		boxprops={'facecolor': 'lightgray', 'edgecolor': 'black'},
		medianprops={'color': 'green'},
		showfliers=not swarm  # hide outliers to reduce overlap with swarm
	)
	if df[metricName].max() > 1:
		ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
		medianStr = "Median: {:,.0f}"
	else:
		ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
		medianStr = "Median: {:,.2f}"
		ax.set_ylim(0, 1)
	ax.grid(which='major', axis='y', linestyle='-', color='lightgray')
	ax.grid(which='minor', axis='y', linestyle=':', color='lightgray')
	plt.xticks(rotation=90)

	if swarm:
		sns.swarmplot(data=df, x=xAxis, y=metricName, size=5, color="green")
	if dfExtra is not False:
		sns.swarmplot(data=dfExtra, x=xAxis, y=metricName, size=8, color="red")

		red_dot = mlines.Line2D(
			[], [], label='Experiment {} to {}'.format(extraTimes[0], extraTimes[-1]),
			color='red', marker='o', linestyle='None', markersize=8, )
		green_dot = mlines.Line2D(
			[], [], label='Baseline {} to {}'.format(mainTimes[0], mainTimes[-1]),
			color='green', marker='o', linestyle='None', markersize=8)
		ax.legend(handles=[red_dot, green_dot], loc='upper left')

	for i, day in enumerate(dayList):
		values = df[df['Day'] == day][metricName].explode().astype(float)
		if not values.empty:
			median = np.median(values)
			ax.text(i + 0.2, median, medianStr.format(median),
				color='black', ha='left', va='center', fontsize=10)

	plt.title("{}. Box plots for baseline only.".format(metricName))
	plt.ylabel(metricName)
	plt.xlabel("Day of Week")
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def PlotWeekStats(weekDaily, weekTimes=False, extraPoints=False, extraTimes=False):
	metricName = "Player Minutes"
	df = pd.DataFrame([
		{"Day": (day - timedelta(hours=12)).strftime('%A'), metricName: pm}
		for day, data in weekDaily.items()
		for pm in data["playerminutesList"]
	])
	dayList = [(day - timedelta(hours=12)).strftime('%A') for day in weekDaily.keys()]
	if extraPoints is not False:
		dfExtra = pd.DataFrame([
			{"Day": (day - timedelta(hours=12)).strftime('%A'), metricName: data['playerminutes']}
			for day, data in extraPoints.items()
		])
	else:
		dfExtra = False
	MakeBoxplot("Day", metricName, dayList, df, weekTimes, dfExtra, extraTimes, True)


def GetMetricDf(
		sizeThreshold, metricName, metricKey,
		weekDaily, weekTimes, extraPoints, extraTimes):
	df = pd.DataFrame([
		{
			"Day": (day - timedelta(hours=12)).strftime('%A'), 
			metricName: CountThresholdProp(dayTimeline, sizeThreshold)
		}
		for day, data in weekDaily.items()
		for dayTimeline in data["{}List".format(metricKey)]
	])
	dayList = [(day - timedelta(hours=12)).strftime('%A') for day in weekDaily.keys()]

	if extraPoints is not False:
		dfExtra = pd.DataFrame([
			{
				"Day": (day - timedelta(hours=12)).strftime('%A'),
				metricName: CountThresholdProp(data[metricKey], sizeThreshold)}
			for day, data in extraPoints.items()
		])
	else:
		dfExtra = False
	return df, dfExtra


def PlotPlayerThresholdUptime(gameSizes, weekDaily, weekTimes=False, extraPoints=False, extraTimes=False):
	metricKey = "dayTimeline"
	metricName = "Proportion of time with at least N players playing across all team games"
	dfList = []
	dfExtraList = []
	for sizeThreshold in gameSizes:
		df, dfExtra = GetMetricDf(
			sizeThreshold, metricName, metricKey, weekDaily, weekTimes, extraPoints, extraTimes)
		df["Size"] = sizeThreshold
		dfExtra["Size"] = sizeThreshold
		dfList.append(df)
		dfExtraList.append(dfExtra)
	df = pd.concat(dfList, ignore_index=True)
	dfExtra = pd.concat(dfExtraList, ignore_index=True)
	xAxis = "DaySize"
	df[xAxis] = df["Day"] + "\nplayers ≥ " + df["Size"].astype(str)
	dfExtra[xAxis] = dfExtra["Day"] + "\nplayers ≥ " + dfExtra["Size"].astype(str)

	dayList = [(day - timedelta(hours=12)).strftime('%A') for day in weekDaily.keys()]
	dayList = ["{}\nplayers ≥ {}".format(day, size) for day in dayList for size in gameSizes]
	MakeBoxplot(xAxis, metricName, dayList, df, weekTimes, dfExtra, extraTimes, False)


def PlotGameSizeUptime(gameSizes, weekDaily, weekTimes=False, extraPoints=False, extraTimes=False):
	metricKey = "dayMax"
	metricName = "Proportion of the day where a team game of size ≥ N is running"
	dfList = []
	dfExtraList = []
	for sizeThreshold in gameSizes:
		df, dfExtra = GetMetricDf(
			sizeThreshold, metricName, metricKey, weekDaily, weekTimes, extraPoints, extraTimes)
		df["Size"] = sizeThreshold
		dfExtra["Size"] = sizeThreshold
		dfList.append(df)
		dfExtraList.append(dfExtra)
	df = pd.concat(dfList, ignore_index=True)
	dfExtra = pd.concat(dfExtraList, ignore_index=True)
	xAxis = "DaySize"
	df[xAxis] = df["Day"] + "\nsize ≥ " + df["Size"].astype(str)
	dfExtra[xAxis] = dfExtra["Day"] + "\nsize ≥ " + dfExtra["Size"].astype(str)

	dayList = [(day - timedelta(hours=12)).strftime('%A') for day in weekDaily.keys()]
	dayList = ["{}\nsize ≥ {}".format(day, size) for day in dayList for size in gameSizes]
	MakeBoxplot(xAxis, metricName, dayList, df, weekTimes, dfExtra, extraTimes, False)


def GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage):
	minTime = False
	maxTime = False
	battleCount = {k : len(v) for k, v in processed.items()}
	mostBattles = heapq.nlargest(trackCount, battleCount, key=battleCount.get)

	print("Get timeline")
	for title in mostBattles:
		print(title)
		battles = processed[title]
		for data in battles:
			if minTime is False or minTime > data["start"]:
				minTime = data["start"]
			if maxTime is False or maxTime < data["end"]:
				maxTime = data["end"]
	
	minTime = minTime.replace(minute=math.floor(minTime.minute/minuteScale)*minuteScale, second=0)
	if weekAverage:
		minTime = minTime.replace(hour=0, minute=0, second=0) + timedelta(hours=dayOffset - 24)
	maxTime = maxTime.replace(minute=math.floor(maxTime.minute/minuteScale)*minuteScale, second=0)
	maxTime = maxTime + timedelta(minutes=minuteScale)
	
	times = []
	time = minTime
	step = timedelta(minutes=minuteScale)
	while time < maxTime:
		times.append(time)
		time = time + step

	averageCounts = MakeAverageCounts(processed, minTime, step, times, mostBattles)
	maxCounts = MakeMaximumCounts(processed, minTime, step, times, mostBattles)
	daily = GetDailyValues(times, maxCounts, averageCounts, minuteScale, dayOffset, mostBattles, processed)
	rawRange = [times[0], times[-1]]
	if weekAverage:
		maxCounts, daily, times = GetWeekAverage(maxCounts, daily, minuteScale, minTime)
	return times, daily, maxCounts, rawRange


def ProcessReplayFiles(files, filterOut):
	processed = {}
	for file in files:
		with open(file) as file:
			lines = []
			for line in file:
				line = line.rstrip()
				if line != "":
					lines.append(line)
					if 'Duration' in line:
						ProcessBlock(lines, processed, filterOut)
						lines = []
	for title, battles in processed.items():
		battles.sort(key=SortBattles)
	return processed


filterOut = [
	"[A] Pro 1v1 Host",
	"[A] Pro 1v1 Host #2",
	"[A] COOP vs AI",
	"[A] Future Wars Coop",
	"[A] Units Level Up",
	"[A] Arena Mod",
	"[A] Free For All",
	"[A] Future Warz",
	"[A] Arena All Welcome",
	"[A] Free For All (No Elo)",
]
minuteScale = 10
dayOffset = 6

def GetBigData(weekAverage):
	trackCount = 10
	processed = ProcessReplayFiles(["big.txt", "janFeb.txt"], filterOut)
	times, daily, counts, rawRange = GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage)
	return times, daily, counts, rawRange


def PlotBigData():
	weekAverage = True
	times, daily, counts, rawRange = GetBigData(weekAverage)
	PlotWeekStats(daily)
	PlotTimeline(times, daily, counts, minuteScale, weekAverage)


def PlotBigDataEveryWeek():
	weekAverage = False
	times, daily, counts, rawRange = GetBigData(weekAverage)
	PlotTimeline(times, daily, counts, minuteScale, weekAverage)


def PlotExperimentData(wantBoxPlot):
	trackCount = 10
	gameUpSizes = [8, 16, 22, 32]
	weekAverage = False
	experimentStart = datetime(2025, 7, 11, 6, 0)
	processed = ProcessReplayFiles(["early2.txt", "customNonsense.txt"], filterOut)
	#MakeBattleList(processed)
	times, daily, counts, rawRange = GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage)
	PlotTimeline(times, daily, counts, minuteScale, weekAverage)
	
	if wantBoxPlot:
		expDaily = {k : v for k, v in daily.items() if k > experimentStart}
		bigTimes, bigDaily, bigCounts, bigRange = GetBigData(True)
		PlotWeekStats(bigDaily, bigRange, expDaily, [experimentStart, rawRange[-1]])
		PlotGameSizeUptime(gameUpSizes, bigDaily, bigRange, expDaily, [experimentStart, rawRange[-1]])
		PlotPlayerThresholdUptime(gameUpSizes, bigDaily, bigRange, expDaily, [experimentStart, rawRange[-1]])

#PlotBigDataEveryWeek()
PlotExperimentData(True)