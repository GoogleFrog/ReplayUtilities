from datetime import datetime, timedelta
import heapq
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
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
	ax.set_xlim(list(dayAgg.values())[0]["end"] - timedelta(hours=24), list(dayAgg.values())[-1]["end"])

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(labels)]
	ax.stackplot(times, values, labels=labels, step='post', colors=colors[::-1])
	ax.plot(times, specs, label='Spectators', linestyle='--', color='black', linewidth=1)
	if weekAverage:
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
	else:
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d/%m'))
	ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

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
	ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.835), fontsize=10)
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
		else:
			match = match[0]
			weekDaily[match]["playerminutes"] += data["playerminutes"]
			weekDaily[match]["battleSizes"] = [
				x + y for (x, y) in zip(weekDaily[match]["battleSizes"], data["battleSizes"])]
			weekDaily[match]["copies"] += 1

	for day, data in weekDaily.items():
		data["playerminutes"] /= data["copies"]
		data["battleSizes"] = [x / data["copies"] for x in data["battleSizes"]]

	return weekCounts, weekDaily, times


def MakeTimeline(processed, trackCount, minuteScale, dayOffset, weekAverage):
	minTime = False
	maxTime = False
	battleCount = {k : len(v) for k, v in processed.items()}
	mostBattles = heapq.nlargest(trackCount, battleCount, key=battleCount.get)
	print(processed.keys())

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
	if weekAverage:
		weekCounts, weekDaily, weekTimes = GetWeekAverage(maxCounts, daily, minuteScale, minTime)
		PlotTimeline(weekTimes, weekDaily, weekCounts, minuteScale, weekAverage)
	else:
		PlotTimeline(times, daily, maxCounts, minuteScale, weekAverage)


def ProcessReplayFiles(files, filterOut):
	for file in files:
		with open(file) as file:
			lines = []
			processed = {}
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

#file = "paste13_06_25_to_13_07_25.txt"
files = ["early.txt"]
trackCount = 6
minuteScale = 10
dayOffset = 6
weekAverage = False

filterOut = [
	"[A] Pro 1v1 Host",
	"[A] COOP vs AI",
	"[A] Future Wars Coop",
	"[A] Units Level Up",
	"[A] Arena Mod",
]

processed = ProcessReplayFiles(files, filterOut)
#MakeBattleList(processed)
MakeTimeline(processed, trackCount, minuteScale, dayOffset, weekAverage)