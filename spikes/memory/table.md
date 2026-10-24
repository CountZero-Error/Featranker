| config | status | input MB (train+eval) | RSS before fit MB | peak RSS MB | peak/input | (peak-before)/input | fit s | rank s (measured) | est. full rank s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| autogroups-10k-c500 | ok | 76.3 | 248 | 576 | 7.5 | 4.3 |  |  |  |
| autogroups-100k-c500 | ok | 762.9 | 939 | 3595 | 4.7 | 3.5 |  |  |  |
| numeric-10k-c20 | ok | 3.1 | 176 | 274 | 88.4 | 31.6 | 3.69 | 8.41 |  |
| mixed-10k-card10 | ok | 9.2 | 182 | 533 | 57.9 | 38.2 | 6.71 | 35.8 |  |
| numeric-10k-c100 | ok | 15.3 | 187 | 440 | 28.8 | 16.5 | 14.01 | 81.53 |  |
| mixed-10k-card10-ms5000 | ok | 9.2 | 184 | 549 | 59.7 | 39.7 | 56.57 | 17.09 |  |
| mixed-10k-card10000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| mixed-10k-card10000-shadows | skipped: dense one-hot estimate 14.9 GB per copy | | | | | | | | |
| mixed-10k-card10000-ms5000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| numeric-10k-c20-shadows | ok | 3.1 | 176 | 326 | 105.2 | 48.4 | 21.43 | 21.54 |  |
| probe-mixed-10k-card10-shadows | ok | 9.2 | 183 | 623 | 67.7 | 47.8 | 72.77 | 1.09 | 131 |
| probe-numeric-10k-c500 | timeout_120s |  |  | 708 (>=, in fit) |  |  |  |  |  |
| probe-mixed-10k-card1000 | timeout_120s |  |  | 6131 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c500-shadows | timeout_120s |  |  | 996 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c100-shadows | ok | 15.3 | 185 | 382 | 25.0 | 12.9 | 69.28 | 2.16 | 433 |
| probe-mixed-10k-card1000-shadows | timeout_120s |  |  | 7377 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c100-evidence | ok | 15.3 | 187 | 548 | 35.8 | 23.6 | 14.49 | 56.45 | 75 |
| probe-mixed-10k-card1000-ms5000 | timeout_120s |  |  | 6203 (>=, in fit) |  |  |  |  |  |
| probe-numeric-10k-c500-evidence | ok | 76.3 | 251 | 1041 | 13.6 | 10.4 | 73.55 | 39.1 | 638 |
| probe-mixed-100k-card10 | ok | 91.6 | 279 | 1978 | 21.6 | 18.5 | 70.92 | 7.06 | 639 |
| probe-numeric-100k-c20 | ok | 30.5 | 206 | 680 | 22.3 | 15.5 | 39.46 | 2.89 | 88 |
| probe-numeric-100k-c20-shadows | ok | 30.5 | 204 | 826 | 27.1 | 20.4 | 60.23 | 4.67 | 188 |
| probe-mixed-100k-card10-shadows | timeout_120s |  |  | 3004 (>=, in rank) |  |  |  |  |  |
| probe-numeric-100k-c20-ms5000 | ok | 30.5 | 205 | 651 | 21.3 | 14.6 | 37.68 | 0.33 | 10 |
| probe-mixed-100k-card10-ms5000 | ok | 91.6 | 276 | 1908 | 20.8 | 17.8 | 71.68 | 3.09 | 280 |
| probe-mixed-100k-card1000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| probe-mixed-10k-card10000 | skipped: dense one-hot estimate 7.5 GB per copy | | | | | | | | |
| probe-mixed-2k-card10000 | ok | 1.8 | 177 | 2667 | 1481.7 | 1383.3 | 55.77 | 0.8 | 72 |
| probe-numeric-100k-c20-shadows-evidence-ms5000 | timeout_180s |  |  | 1042 (>=, in rank) |  |  |  |  |  |
| probe-mixed-2k-card10000-shadows | timeout_120s |  |  | 3968 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100 | timeout_120s |  |  | 1013 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100-shadows | timeout_120s |  |  | 1925 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c100-ms5000 | ok | 152.6 | 328 | 1352 | 8.9 | 6.7 | 112.12 | 0.35 | 53 |
| probe-numeric-100k-c500 | timeout_120s |  |  | 3555 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c500-ms5000 | timeout_120s |  |  | 3921 (>=, in fit) |  |  |  |  |  |
| probe-numeric-100k-c500-shadows-evidence-ms5000 | timeout_180s |  |  | 4829 (>=, in fit) |  |  |  |  |  |
| probe-numeric-500k-c20 | timeout_120s |  |  | 1630 (>=, in fit) |  |  |  |  |  |
| probe-numeric-500k-c20-ms5000 | timeout_120s |  |  | 1671 (>=, in fit) |  |  |  |  |  |
| overhead-numeric-100k-c100 | ok | 152.6 | 328 | 710 | 4.7 | 2.5 | 0.19 | 0.23 |  |
| overhead-numeric-100k-c100-shadows | ok | 152.6 | 327 | 1214 | 8.0 | 5.8 | 0.2 | 0.87 |  |
| overhead-numeric-100k-c500 | ok | 762.9 | 939 | 2707 | 3.5 | 2.3 | 0.76 | 1.01 |  |
| overhead-numeric-100k-c500-shadows | ok | 762.9 | 937 | 4664 | 6.1 | 4.9 | 1.67 | 2.9 |  |
| overhead-numeric-100k-c500-ms5000 | ok | 762.9 | 936 | 2033 | 2.7 | 1.4 | 0.54 | 0.42 |  |
| overhead-numeric-100k-c500-shadows-ms5000 | ok | 762.9 | 940 | 3292 | 4.3 | 3.1 | 1.08 | 0.64 |  |
| overhead-numeric-100k-c500-shadows-evidence-ms5000 | ok | 762.9 | 939 | 5933 | 7.8 | 6.5 | 1.03 | 6.64 |  |
| overhead-numeric-100k-c500-evidence | ok | 762.9 | 937 | 4803 | 6.3 | 5.1 | 0.7 | 7.99 |  |
| overhead-numeric-500k-c20 | ok | 152.6 | 341 | 846 | 5.5 | 3.3 | 0.25 | 0.62 |  |
| overhead-numeric-500k-c20-shadows | ok | 152.6 | 353 | 1330 | 8.7 | 6.4 | 0.32 | 0.96 |  |
| overhead-mixed-100k-card10 | ok | 91.6 | 287 | 779 | 8.5 | 5.4 | 4.54 | 2.94 |  |
| overhead-mixed-100k-card10-shadows | ok | 91.6 | 286 | 1062 | 11.6 | 8.5 | 3.2 | 3.73 |  |
| overhead-mixed-10k-card1000 | ok | 9.2 | 183 | 242 | 26.3 | 6.4 | 0.27 | 0.24 |  |
| overhead-mixed-10k-card1000-shadows | ok | 9.2 | 184 | 296 | 32.2 | 12.2 | 0.28 | 0.24 |  |
