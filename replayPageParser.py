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
		ax.axvline(x=dayData["end"], color='black', linestyle='--', linewidth=0.5)
		ax.text(dayData["start"] + timedelta(hours=2), ax.get_ylim()[1] - 1,
			"{}\n{:,.0f} pm\n{:,.0f} games\n{:,.0f} with ≥ 16\n{:,.0f} with ≥ 22\n{:,.0f} with ≤ 10".format(
				(dayData["end"] - timedelta(hours=12)).strftime('%A'),
				dayData["playerminutes"],
				sum(dayData["battleSizes"]),
				sum(dayData["battleSizes"][16::]),
				sum(dayData["battleSizes"][22::]),
				sum(dayData["battleSizes"][::10])),
			rotation=0, verticalalignment='top', horizontalalignment='left',
			backgroundcolor='white', fontsize=10)

	handles = [Patch(facecolor=colors[i], label=label) for i, label in enumerate(labels[::-1])]
	line_handle = Line2D([0], [0], color='black', linewidth=2, label='Spectators (smoothed)')
	handles.append(line_handle)
	ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.835), fontsize=7)
	if weekAverage:
		title = "Average players in team games"
	else:
		title = "Maximum players in team games"
	plt.title(
		'{} by {}-minute period {} to {}. Counts in playerminutes (pm). Game counts include ≥ 5 minutes.'.format(
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

def GetDailyValues(times, data, step, offset, mostBattles, processed):
	data = data.copy()
	del data["specs"]

	outputs = {}
	nextDay = GetDayEnd(times[0], offset)
	lastTime = times[0]
	index = 0
	playerAcc = 0
	gameAcc = 0
	for time in times:
		if time >= nextDay:
			outputs[nextDay] = {
				"start" : lastTime,
				"end" : nextDay, 
				"playerminutes" : playerAcc*step,
				"battleSizes" : [0] * 40,
			}
			nextDay = GetDayEnd(time, offset)
			lastTime = time
			playerAcc = 0
		for count in data.values():
			playerAcc += count[index]
		index += 1
	
	outputs[nextDay] = {
		"start" : lastTime,
		"end" : time, 
		"playerminutes" : playerAcc*step,
		"battleSizes" : [0] * 40,
	}

	for title in mostBattles:
		battles = processed[title]
		for data in battles:
			battleDay = GetDayEnd(data["start"], offset)
			if battleDay in outputs:
				if data["duration"] >= 5:
					outputs[battleDay]["battleSizes"][data["players"]] += 1
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
		else:
			match = match[0]
			weekDaily[match]["playerminutes"] += data["playerminutes"]
			weekDaily[match]["playerminutesList"].append(data["playerminutes"])
			weekDaily[match]["battleSizes"] = [
				x + y for (x, y) in zip(weekDaily[match]["battleSizes"], data["battleSizes"])]
			weekDaily[match]["copies"] += 1

	for day, data in weekDaily.items():
		data["playerminutes"] /= data["copies"]
		data["battleSizes"] = [x / data["copies"] for x in data["battleSizes"]]
		dist = data["playerminutesList"]

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


def PlotWeekStats(weekDaily, weekTimes=False, extraPoints=False, extraTimes=False):
	df = pd.DataFrame([
		{"Day": (day - timedelta(hours=12)).strftime('%A'), "Player Minutes": pm}
		for day, data in weekDaily.items()
		for pm in data["playerminutesList"]
	])
	dayList = [(day - timedelta(hours=12)).strftime('%A') for day in weekDaily.keys()]
	# Create the swarm plot
	plt.figure(figsize=(10, 6))
	ax = sns.boxplot(
		data=df,
		x="Day",
		y="Player Minutes",
		order=dayList,
		width=0.3,
		showcaps=True,
		boxprops={'facecolor': 'lightgray', 'edgecolor': 'black'},
		medianprops={'color': 'red'},
		showfliers=False  # hide outliers to reduce overlap with swarm
	)
	ax.yaxis.set_major_locator(ticker.MultipleLocator(5000))
	ax.yaxis.set_minor_locator(ticker.MultipleLocator(1000))
	ax.grid(which='major', axis='y', linestyle='-', color='lightgray')
	ax.grid(which='minor', axis='y', linestyle=':', color='lightgray')

	sns.swarmplot(data=df, x="Day", y="Player Minutes", size=6, color="green")
	if extraPoints is not False:
		dfExtra = pd.DataFrame([
			{"Day": (day - timedelta(hours=12)).strftime('%A'), "Player Minutes": data['playerminutes']}
			for day, data in extraPoints.items()
		])
		sns.swarmplot(data=dfExtra, x="Day", y="Player Minutes", size=8, color="red")

		red_dot = mlines.Line2D(
			[], [], label='Experiment {} to {}'.format(extraTimes[0], extraTimes[-1]),
			color='red', marker='o', linestyle='None', markersize=8, )
		green_dot = mlines.Line2D(
			[], [], label='Baseline {} to {}'.format(weekTimes[0], weekTimes[-1]),
			color='green', marker='o', linestyle='None', markersize=8)
		ax.legend(handles=[red_dot, green_dot], loc='upper left')

	for i, day in enumerate(dayList):
		values = df[df['Day'] == day]['Player Minutes'].explode().astype(float)
		if not values.empty:
			median = np.median(values)
			ax.text(i + 0.2, median, "Median: {:,.0f}".format(median),
				color='black', ha='left', va='center', fontsize=10)

	plt.title("Player Minutes Distribution by Day. Box plots for baseline only.")
	plt.ylabel("Player Minutes")
	plt.xlabel("Day of Week")
	plt.grid(True)
	plt.tight_layout()
	plt.show()


def GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage):
	minTime = False
	maxTime = False
	battleCount = {k : len(v) for k, v in processed.items()}
	mostBattles = heapq.nlargest(trackCount, battleCount, key=battleCount.get)
	#print(processed.keys())

	for title in mostBattles:
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
	daily = GetDailyValues(times, averageCounts, minuteScale, dayOffset, mostBattles, processed)
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
	"[A] COOP vs AI",
	"[A] Future Wars Coop",
	"[A] Units Level Up",
	"[A] Arena Mod",
]
minuteScale = 10
dayOffset = 6

def GetBigData():
	trackCount = 5
	weekAverage = True
	processed = ProcessReplayFiles(["big.txt", "janFeb.txt"], filterOut)
	times, daily, counts, rawRange = GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage)
	return times, daily, counts, rawRange


def PlotBigData():
	weekAverage = True
	times, daily, counts, rawRange = GetBigData()
	PlotWeekStats(daily)
	PlotTimeline(times, daily, counts, minuteScale, weekAverage)


def PlotExperimentData():
	trackCount = 7
	weekAverage = False
	experimentStart = datetime(2025, 7, 11, 6, 0)
	processed = ProcessReplayFiles(["early.txt"], filterOut)
	times, daily, counts, rawRange = GetTimelineData(processed, trackCount, minuteScale, dayOffset, weekAverage)
	PlotTimeline(times, daily, counts, minuteScale, weekAverage)
	
	expDaily = {k : v for k, v in daily.items() if k > experimentStart}
	bigTimes, bigDaily, _, bigRange = GetBigData()
	PlotWeekStats(bigDaily, bigRange, expDaily, [experimentStart, rawRange[-1]])

PlotExperimentData()