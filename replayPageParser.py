from datetime import datetime, timedelta
import heapq
import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import pandas as pd

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


def SortBattles(a):
	return a["start"]


def MakeBattleList(processed):
	print('{},{},{},{},{}'.format("Room", "Players", "Time", "Duration", "Spectators"))
	for title, battles in processed.items():
		battles.sort(key=SortBattles)
		for data in battles:
			print('{},{},{},{},{}'.format(
				title, data["players"], data["start"].strftime("%H:%M"),
				data["duration"], data["specs"]))


def PlotTimeline(times, data, minuteScale):
	data = data.copy()
	del data["specs"]

	labels = list(data.keys())[::-1]
	values = [data[label] for label in labels]

	fig, ax = plt.subplots(figsize=(12, 6))

	ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
	ax.grid(axis='y', linestyle='-', color='lightgray')
	ax.grid(axis='x', linestyle='-', color='lightgray')
	plt.axhline(y=32, linestyle='-', color='red')
	ax.set_axisbelow(True)

	colors = plt.rcParams['axes.prop_cycle'].by_key()['color'][:len(labels)]
	ax.stackplot(times, values, labels=labels, step='post', colors=colors[::-1])
	ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %d/%m'))
	ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

	handles = [Patch(facecolor=colors[i], label=label) for i, label in enumerate(labels[::-1])]
	ax.legend(handles=handles, loc='upper left')
	plt.title('Maximum players in team games by {}-minute period {} to {}'.format(minuteScale, times[0], times[-1]))
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


def PrintDailyValues(times, data, step, offset):
	data = data.copy()
	del data["specs"]

	outputs = []
	lastTime = times[0]
	index = 0
	acc = 0
	for time in times:
		if (time - timedelta(hours=offset)).date() != (lastTime - timedelta(hours=offset)).date():
			outputs.append((lastTime, acc))
			lastTime = time
			acc = 0
		for count in data.values():
			acc += count[index]
		index += 1
	
	for value in outputs:
		print('{}-{}: {}'.format(
			value[0].strftime("%I:%M %d/%m %p"),
			(value[0]+ timedelta(hours=24)).strftime("%I:%M %d/%m %p"),
			value[1]*step
		))


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


def MakeTimeline(processed, trackCount, minuteScale):
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

	PrintDailyValues(times, averageCounts, minuteScale, 6)
	#PrintTimeline(minTime, maxTime, minuteScale, maxCounts)
	PlotTimeline(times, maxCounts, minuteScale)


#file = "paste13_06_25_to_13_07_25.txt"
file = "early.txt"
trackCount = 5
minuteScale = 10

filterOut = [
	"[A] Pro 1v1 Host",
	"[A] COOP vs AI",
	"[A] Future Wars Coop",
	"[A] Units Level Up",
	"[A] Arena Mod",
]

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
	
	#MakeBattleList(processed)
	MakeTimeline(processed, trackCount, minuteScale)