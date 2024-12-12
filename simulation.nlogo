;; Global variables
globals [
  average-wealth
  group-a-avg-wealth
  group-b-avg-wealth
  num-group-a
  num-group-b
  taxes-rate
  export-data
]

;; Agent properties
turtles-own [
  wealth
  group           ;; "A" or "B"
  opportunities   ;; true/false
  career-years
  sex            ;; "M" or "F"
  has-job?
  age
  age-of-death
  num-diseases
  has-disease?
  has-car?
  has-house?
  job-loss-probability
  reproduction-chance
  child-possibility
  num-children
  personal-luxuries?
  health-care-cost
]

;; Setup
to setup
  clear-all
  setup-agents
  setup-plots
  set taxes-rate 0.1 ;; 10% tax rate
  reset-ticks
  ask turtles [
  set-entity-image group sex
]

end

to setup-agents
  create-turtles num-agents-a [
    set group "A"
    set wealth random-float 5 + 5  ;; 5-10 range
    set opportunities random-float 1 < 0.8  ;; 80% chance
    set sex ifelse-value (random-float 1 < 0.5) ["M"] ["F"]
    setup-common-properties "A"
    set color blue
  ]

  create-turtles num-agents-b [
    set group "B"
    set wealth random-float 4 + 1  ;; 1-5 range
    set opportunities random-float 1 < 0.3  ;; 30% chance
    set sex ifelse-value (random-float 1 < 0.6) ["M"] ["F"]
    setup-common-properties "B"
    set color red
  ]

  ask turtles [
    setxy random-xcor random-ycor
  ]
end

to setup-common-properties [g]
  set has-job? true
  set age 0
  set age-of-death ifelse-value (g = "A") [age-of-death-a] [age-of-death-b]
  set num-diseases 0
  set has-disease? false
  set has-car? false
  set has-house? false
  set job-loss-probability ifelse-value (g = "A") [0.05] [0.15]
  set reproduction-chance 0.05
  set child-possibility 1
  set num-children 0
  set personal-luxuries? false
  set health-care-cost 0.3
  set career-years 0
  set-entity-image g sex
end

to set-entity-image [entity-group entity-sex]
  ifelse group = "A"
    [ ifelse sex = "M"
      [ set shape "male" ]
      [ set shape "female" ]
    ]
    [ ifelse sex = "M"
      [ set shape "male" ]
      [ set shape "female" ]
    ]
  set size 1.5  ; Adjust size as needed
end


to update-statistics
  ;; Check if there are any turtles before calculating means
  ifelse any? turtles [
    set average-wealth mean [wealth] of turtles

    ;; Check for Group A turtles
    ifelse any? turtles with [group = "A"] [
      set group-a-avg-wealth mean [wealth] of turtles with [group = "A"]
    ] [
      set group-a-avg-wealth 0
    ]

    ;; Check for Group B turtles
    ifelse any? turtles with [group = "B"] [
      set group-b-avg-wealth mean [wealth] of turtles with [group = "B"]
    ] [
      set group-b-avg-wealth 0
    ]

    ;; Update population counts
    set num-group-a count turtles with [group = "A"]
    set num-group-b count turtles with [group = "B"]
  ] [
    ;; If no turtles exist, set all values to 0
    set average-wealth 0
    set group-a-avg-wealth 0
    set group-b-avg-wealth 0
    set num-group-a 0
    set num-group-b 0
  ]
end

to update-our-plots
  set-current-plot "Population Groups"
  set-current-plot-pen "Group A"
  plot num-group-a
  set-current-plot-pen "Group B"
  plot num-group-b

  set-current-plot "Wealth Distribution"
  set-current-plot-pen "Avg Wealth A"
  plot group-a-avg-wealth
  set-current-plot-pen "Avg Wealth B"
  plot group-b-avg-wealth
end

;; Main simulation step
to go
  if ticks >= max-steps [
    export-to-csv
    stop
  ]

  ask turtles [
    age-and-health
    if age >= age-of-death [
      die
    ]

    if age < age-of-death [
      wealth-dynamics
      check-assets
      move-and-interact
      reproduce
    ]

    set size  size + 0.001 * wealth
  ]

  update-statistics
  update-our-plots
  export-to-csv
  tick
end

;; Agent procedures
to age-and-health
  set age age + 1

  if wealth > (0.5 * mean [wealth] of turtles) [
    set age-of-death age-of-death + 0.65
  ]

  let healthcare-cost (0.4 * age + num-diseases * 0.2)
  if wealth >= healthcare-cost [
    set wealth wealth - healthcare-cost
  ]

  if random-float 1 < (ifelse-value group = "A" [0.01] [0.05]) [
    if num-diseases = 0 [
      set has-disease? true
      set num-diseases num-diseases + 1
      set age-of-death age-of-death - 0.2
    ]
  ]
end

to wealth-dynamics
  if has-job? and age >= 18 [
    set career-years career-years + 1
    let growth-rate ifelse-value (group = "A") [group-a-wealth-rate] [group-b-wealth-rate]

    if sex = "F" [
      set growth-rate growth-rate * 0.7
    ]

    if opportunities [
      set growth-rate growth-rate * 1.5
    ]

    set wealth wealth + random-float growth-rate

     ;; Apply taxes for turtles aged 18+
    let taxes wealth * taxes-rate
    set wealth wealth - taxes

    if random-float 1 < job-loss-probability [
      set has-job? false
    ]
  ]
  if wealth < 0 [ set wealth 0 ]
end

to check-assets
  let avg-wealth mean [wealth] of turtles
  if wealth < 0 [ set wealth 0 ]
  if wealth > (0.7 * avg-wealth) and not has-car? [
    set has-car? true
    set wealth wealth * 0.7
    set reproduction-chance reproduction-chance * 3
    set job-loss-probability job-loss-probability / 2
  ]

  if wealth > (0.9 * avg-wealth) and not has-house? [
    set has-house? true
    set wealth wealth * 0.3
    set job-loss-probability job-loss-probability / 4
  ]
end

to move-and-interact
  rt random 360
  fd 1

  let nearby-agent one-of other turtles-here
  if nearby-agent != nobody [
    if wealth > [wealth] of nearby-agent [
      let transfer random-float 0.15 * wealth
      set wealth wealth + transfer
      ask nearby-agent [ set wealth wealth - transfer ]
    ]
  ]
end

to reproduce
  if age >= 18 and num-children < 3 [
    let potential-mate one-of other turtles-here with [
      sex != [sex] of myself and
      group = [group] of myself and
      age >= 18 and
      num-children < 3
    ]

    if potential-mate != nobody and random-float 1 < reproduction-chance [
      hatch 1 [
        set age 0
        set wealth random-float 5
        set sex ifelse-value (random-float 1 < 0.5) ["M"] ["F"]
        setup-common-properties [group] of myself
        setxy random-xcor random-ycor
      ]
      set num-children num-children + 1
      ask potential-mate [
        set num-children num-children + 1
      ]
    ]
  ]
end

to export-to-csv
  ifelse (ticks = 0) [
    ; If it's the first tick, create the file and write the header
    if file-exists? "simulation_results.csv" [
      file-delete "simulation_results.csv"
    ]
    file-open "simulation_results.csv"
    file-print (word "tick,agent_id,group,wealth,opportunities,career_years,sex,"
                     "age,has_disease,num_diseases,has_car,has_house,job_status,"
                     "reproduction_chance,num_children,personal_luxuries,health_care_cost")
  ] [
    ; For subsequent ticks, just open the file in append mode
    file-open "simulation_results.csv"
  ]

  ; Write data for each agent
  ask turtles [
    file-print (word ticks "," who "," group "," wealth "," opportunities ","
                     career-years "," sex "," age "," has-disease? "," num-diseases ","
                     has-car? "," has-house? "," has-job? "," reproduction-chance ","
                     num-children "," personal-luxuries? "," health-care-cost)
  ]
  file-close
end
@#$#@#$#@
GRAPHICS-WINDOW
215
9
740
535
-1
-1
15.67
1
10
1
1
1
0
0
0
1
-16
16
-16
16
1
1
1
ticks
30.0

SLIDER
21
22
193
55
num-agents-a
num-agents-a
0
500
250.0
1
1
NIL
HORIZONTAL

SLIDER
21
67
193
100
num-agents-b
num-agents-b
0
500
250.0
1
1
NIL
HORIZONTAL

SLIDER
22
124
194
157
age-of-death-a
age-of-death-a
0
100
90.0
1
1
NIL
HORIZONTAL

SLIDER
23
168
195
201
age-of-death-b
age-of-death-b
0
100
80.0
1
1
NIL
HORIZONTAL

SLIDER
21
312
193
345
max-steps
max-steps
0
500
200.0
10
1
NIL
HORIZONTAL

SLIDER
23
222
195
255
group-a-wealth-rate
group-a-wealth-rate
0
1
0.6
0.1
1
NIL
HORIZONTAL

SLIDER
23
266
195
299
group-b-wealth-rate
group-b-wealth-rate
0
1
0.3
0.1
1
NIL
HORIZONTAL

BUTTON
848
126
911
159
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
848
76
911
109
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

PLOT
954
29
1308
261
Population groups
Time
Number of Agents
0.0
100.0
0.0
500.0
true
true
"" ""
PENS
"Group A" 1.0 0 -13791810 true "" ""
"Group B" 1.0 0 -5298144 true "" ""

PLOT
954
278
1310
519
Wealth Distribution
Time
Average Wealth
0.0
100.0
0.0
20.0
true
true
"" ""
PENS
"Avg Wealth A" 1.0 0 -13791810 true "" ""
"Avg Wealth B" 1.0 0 -5298144 true "" ""

MONITOR
784
301
933
346
Average Wealth Group A
group-a-avg-wealth
3
1
11

MONITOR
785
356
931
401
Average Wealth Group B
group-b-avg-wealth
3
1
11

MONITOR
813
412
934
457
Population Group A
num-group-a
0
1
11

MONITOR
812
464
933
509
Population Group B
num-group-b
0
1
11

BUTTON
848
29
913
62
Export
export-to-csv
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

@#$#@#$#@
## WHAT IS IT?

This model simulates a society with two distinct groups (A and B) to explore wealth dynamics, social mobility, and demographic changes over time. It aims to demonstrate how initial advantages or disadvantages can impact long-term outcomes for individuals and groups.

## HOW IT WORKS

The model creates two groups of agents with different initial wealth distributions and opportunities:

- Group A: Higher initial wealth (5-10 range) and better opportunities (80% chance)
- Group B: Lower initial wealth (1-5 range) and fewer opportunities (30% chance)

Agents have various properties including wealth, job status, health, assets, and family status. The simulation progresses through the following main processes:

1. Aging and health management
2. Wealth dynamics and career progression
3. Asset acquisition (cars and houses)
4. Social interaction and wealth transfer
5. Reproduction

## HOW TO USE IT

1. Set the initial parameters:
   - Number of agents in Group A and Group B
   - Wealth growth rates for both groups
   - Maximum simulation steps

2. Click the "Setup" button to initialize the model.

3. Click the "Go" button to run the simulation.

4. Observe the changes in population and wealth distribution through the provided plots.

## THINGS TO NOTICE

- How do the populations of Group A and Group B change over time?
- What are the differences in average wealth between the two groups?
- How do factors like job opportunities, health, and asset ownership affect an agent's wealth trajectory?

## THINGS TO TRY

- Adjust the initial wealth ranges and opportunity percentages for each group
- Modify the wealth growth rates to see how they impact long-term inequality
- Change the reproduction chances and see how it affects population dynamics

## RELATED MODELS

- Schelling Segregation Model
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

airplane
true
0
Polygon -7500403 true true 150 0 135 15 120 60 120 105 15 165 15 195 120 180 135 240 105 270 120 285 150 270 180 285 210 270 165 240 180 180 285 195 285 165 180 105 180 60 165 15

arrow
true
0
Polygon -7500403 true true 150 0 0 150 105 150 105 293 195 293 195 150 300 150

box
false
0
Polygon -7500403 true true 150 285 285 225 285 75 150 135
Polygon -7500403 true true 150 135 15 75 150 15 285 75
Polygon -7500403 true true 15 75 15 225 150 285 150 135
Line -16777216 false 150 285 150 135
Line -16777216 false 150 135 15 75
Line -16777216 false 150 135 285 75

bug
true
0
Circle -7500403 true true 96 182 108
Circle -7500403 true true 110 127 80
Circle -7500403 true true 110 75 80
Line -7500403 true 150 100 80 30
Line -7500403 true 150 100 220 30

butterfly
true
0
Polygon -7500403 true true 150 165 209 199 225 225 225 255 195 270 165 255 150 240
Polygon -7500403 true true 150 165 89 198 75 225 75 255 105 270 135 255 150 240
Polygon -7500403 true true 139 148 100 105 55 90 25 90 10 105 10 135 25 180 40 195 85 194 139 163
Polygon -7500403 true true 162 150 200 105 245 90 275 90 290 105 290 135 275 180 260 195 215 195 162 165
Polygon -16777216 true false 150 255 135 225 120 150 135 120 150 105 165 120 180 150 165 225
Circle -16777216 true false 135 90 30
Line -16777216 false 150 105 195 60
Line -16777216 false 150 105 105 60

car
false
0
Polygon -7500403 true true 300 180 279 164 261 144 240 135 226 132 213 106 203 84 185 63 159 50 135 50 75 60 0 150 0 165 0 225 300 225 300 180
Circle -16777216 true false 180 180 90
Circle -16777216 true false 30 180 90
Polygon -16777216 true false 162 80 132 78 134 135 209 135 194 105 189 96 180 89
Circle -7500403 true true 47 195 58
Circle -7500403 true true 195 195 58

circle
false
0
Circle -7500403 true true 0 0 300

circle 2
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240

cow
false
0
Polygon -7500403 true true 200 193 197 249 179 249 177 196 166 187 140 189 93 191 78 179 72 211 49 209 48 181 37 149 25 120 25 89 45 72 103 84 179 75 198 76 252 64 272 81 293 103 285 121 255 121 242 118 224 167
Polygon -7500403 true true 73 210 86 251 62 249 48 208
Polygon -7500403 true true 25 114 16 195 9 204 23 213 25 200 39 123

cylinder
false
0
Circle -7500403 true true 0 0 300

dot
false
0
Circle -7500403 true true 90 90 120

face happy
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 255 90 239 62 213 47 191 67 179 90 203 109 218 150 225 192 218 210 203 227 181 251 194 236 217 212 240

face neutral
false
0
Circle -7500403 true true 8 7 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Rectangle -16777216 true false 60 195 240 225

face sad
false
0
Circle -7500403 true true 8 8 285
Circle -16777216 true false 60 75 60
Circle -16777216 true false 180 75 60
Polygon -16777216 true false 150 168 90 184 62 210 47 232 67 244 90 220 109 205 150 198 192 205 210 220 227 242 251 229 236 206 212 183

female
false
0
Circle -7500403 true true 110 5 80
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 105 90 60 150 60 195 120 120
Polygon -7500403 true true 120 165 135 210
Polygon -7500403 true true 120 90 195 90 225 255 75 255 105 90
Rectangle -7500403 true true 105 255 135 300
Rectangle -7500403 true true 165 255 195 300
Polygon -7500403 true true 195 90 240 150 240 195 180 120

fish
false
0
Polygon -1 true false 44 131 21 87 15 86 0 120 15 150 0 180 13 214 20 212 45 166
Polygon -1 true false 135 195 119 235 95 218 76 210 46 204 60 165
Polygon -1 true false 75 45 83 77 71 103 86 114 166 78 135 60
Polygon -7500403 true true 30 136 151 77 226 81 280 119 292 146 292 160 287 170 270 195 195 210 151 212 30 166
Circle -16777216 true false 215 106 30

flag
false
0
Rectangle -7500403 true true 60 15 75 300
Polygon -7500403 true true 90 150 270 90 90 30
Line -7500403 true 75 135 90 135
Line -7500403 true 75 45 90 45

flower
false
0
Polygon -10899396 true false 135 120 165 165 180 210 180 240 150 300 165 300 195 240 195 195 165 135
Circle -7500403 true true 85 132 38
Circle -7500403 true true 130 147 38
Circle -7500403 true true 192 85 38
Circle -7500403 true true 85 40 38
Circle -7500403 true true 177 40 38
Circle -7500403 true true 177 132 38
Circle -7500403 true true 70 85 38
Circle -7500403 true true 130 25 38
Circle -7500403 true true 96 51 108
Circle -16777216 true false 113 68 74
Polygon -10899396 true false 189 233 219 188 249 173 279 188 234 218
Polygon -10899396 true false 180 255 150 210 105 210 75 240 135 240

house
false
0
Rectangle -7500403 true true 45 120 255 285
Rectangle -16777216 true false 120 210 180 285
Polygon -7500403 true true 15 120 150 15 285 120
Line -16777216 false 30 120 270 120

leaf
false
0
Polygon -7500403 true true 150 210 135 195 120 210 60 210 30 195 60 180 60 165 15 135 30 120 15 105 40 104 45 90 60 90 90 105 105 120 120 120 105 60 120 60 135 30 150 15 165 30 180 60 195 60 180 120 195 120 210 105 240 90 255 90 263 104 285 105 270 120 285 135 240 165 240 180 270 195 240 210 180 210 165 195
Polygon -7500403 true true 135 195 135 240 120 255 105 255 105 285 135 285 165 240 165 195

line
true
0
Line -7500403 true 150 0 150 300

line half
true
0
Line -7500403 true 150 0 150 150

male
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

pentagon
false
0
Polygon -7500403 true true 150 15 15 120 60 285 240 285 285 120

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

plant
false
0
Rectangle -7500403 true true 135 90 165 300
Polygon -7500403 true true 135 255 90 210 45 195 75 255 135 285
Polygon -7500403 true true 165 255 210 210 255 195 225 255 165 285
Polygon -7500403 true true 135 180 90 135 45 120 75 180 135 210
Polygon -7500403 true true 165 180 165 210 225 180 255 120 210 135
Polygon -7500403 true true 135 105 90 60 45 45 75 105 135 135
Polygon -7500403 true true 165 105 165 135 225 105 255 45 210 60
Polygon -7500403 true true 135 90 120 45 150 15 180 45 165 90

sheep
false
15
Circle -1 true true 203 65 88
Circle -1 true true 70 65 162
Circle -1 true true 150 105 120
Polygon -7500403 true false 218 120 240 165 255 165 278 120
Circle -7500403 true false 214 72 67
Rectangle -1 true true 164 223 179 298
Polygon -1 true true 45 285 30 285 30 240 15 195 45 210
Circle -1 true true 3 83 150
Rectangle -1 true true 65 221 80 296
Polygon -1 true true 195 285 210 285 210 240 240 210 195 210
Polygon -7500403 true false 276 85 285 105 302 99 294 83
Polygon -7500403 true false 219 85 210 105 193 99 201 83

square
false
0
Rectangle -7500403 true true 30 30 270 270

square 2
false
0
Rectangle -7500403 true true 30 30 270 270
Rectangle -16777216 true false 60 60 240 240

star
false
0
Polygon -7500403 true true 151 1 185 108 298 108 207 175 242 282 151 216 59 282 94 175 3 108 116 108

target
false
0
Circle -7500403 true true 0 0 300
Circle -16777216 true false 30 30 240
Circle -7500403 true true 60 60 180
Circle -16777216 true false 90 90 120
Circle -7500403 true true 120 120 60

tree
false
0
Circle -7500403 true true 118 3 94
Rectangle -6459832 true false 120 195 180 300
Circle -7500403 true true 65 21 108
Circle -7500403 true true 116 41 127
Circle -7500403 true true 45 90 120
Circle -7500403 true true 104 74 152

triangle
false
0
Polygon -7500403 true true 150 30 15 255 285 255

triangle 2
false
0
Polygon -7500403 true true 150 30 15 255 285 255
Polygon -16777216 true false 151 99 225 223 75 224

truck
false
0
Rectangle -7500403 true true 4 45 195 187
Polygon -7500403 true true 296 193 296 150 259 134 244 104 208 104 207 194
Rectangle -1 true false 195 60 195 105
Polygon -16777216 true false 238 112 252 141 219 141 218 112
Circle -16777216 true false 234 174 42
Rectangle -7500403 true true 181 185 214 194
Circle -16777216 true false 144 174 42
Circle -16777216 true false 24 174 42
Circle -7500403 false true 24 174 42
Circle -7500403 false true 144 174 42
Circle -7500403 false true 234 174 42

turtle
true
0
Polygon -10899396 true false 215 204 240 233 246 254 228 266 215 252 193 210
Polygon -10899396 true false 195 90 225 75 245 75 260 89 269 108 261 124 240 105 225 105 210 105
Polygon -10899396 true false 105 90 75 75 55 75 40 89 31 108 39 124 60 105 75 105 90 105
Polygon -10899396 true false 132 85 134 64 107 51 108 17 150 2 192 18 192 52 169 65 172 87
Polygon -10899396 true false 85 204 60 233 54 254 72 266 85 252 107 210
Polygon -7500403 true true 119 75 179 75 209 101 224 135 220 225 175 261 128 261 81 224 74 135 88 99

wheel
false
0
Circle -7500403 true true 3 3 294
Circle -16777216 true false 30 30 240
Line -7500403 true 150 285 150 15
Line -7500403 true 15 150 285 150
Circle -7500403 true true 120 120 60
Line -7500403 true 216 40 79 269
Line -7500403 true 40 84 269 221
Line -7500403 true 40 216 269 79
Line -7500403 true 84 40 221 269

wolf
false
0
Polygon -16777216 true false 253 133 245 131 245 133
Polygon -7500403 true true 2 194 13 197 30 191 38 193 38 205 20 226 20 257 27 265 38 266 40 260 31 253 31 230 60 206 68 198 75 209 66 228 65 243 82 261 84 268 100 267 103 261 77 239 79 231 100 207 98 196 119 201 143 202 160 195 166 210 172 213 173 238 167 251 160 248 154 265 169 264 178 247 186 240 198 260 200 271 217 271 219 262 207 258 195 230 192 198 210 184 227 164 242 144 259 145 284 151 277 141 293 140 299 134 297 127 273 119 270 105
Polygon -7500403 true true -1 195 14 180 36 166 40 153 53 140 82 131 134 133 159 126 188 115 227 108 236 102 238 98 268 86 269 92 281 87 269 103 269 113

x
false
0
Polygon -7500403 true true 270 75 225 30 30 225 75 270
Polygon -7500403 true true 30 75 75 30 270 225 225 270
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180
@#$#@#$#@
0
@#$#@#$#@
