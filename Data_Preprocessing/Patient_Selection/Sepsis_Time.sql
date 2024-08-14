-- Please import sofa.csv into mysql's mimic3_sepsis_cleaned before running the code.

CREATE TABLE mimic3_sepsis_final.SOFA_TIME
SELECT icustayid,charttime, charttime - 172800 AS starttime , charttime + 86400 AS endtime ,sofa
FROM mimic3_sepsis_cleaned.SOFA
WHERE sofa > 2

-- Integration of total duration of illness
CREATE TABLE SOFA_PERIOD
WITH merged_intervals AS (
    SELECT
        icustayid,
        MIN(starttime) AS starttime,
        MAX(endtime) AS endtime,
        MAX(sofa) AS MAX_SOFA
    FROM (
        SELECT
            t1.icustayid,
            t1.starttime,
            t1.endtime,
            SUM(flag) OVER (PARTITION BY t1.icustayid ORDER BY t1.starttime) AS grp
        FROM (
            SELECT
                icustayid,
                starttime,
                endtime,
                CASE
                    WHEN starttime <= lag(endtime) OVER (PARTITION BY icustayid ORDER BY starttime) THEN 0
                    ELSE 1
                END AS flag
            FROM mimic3_sepsis_final.SOFA_TIME
        ) AS t1
    ) AS t2
    GROUP BY icustayid, grp
)
SELECT
    ICUSTAYID,
    STARTTIME,
    ENDTIME,
    MAX_SOFA,
    ROW_NUMBER() OVER (PARTITION BY icustayid ORDER BY starttime) AS NUMBER
FROM merged_intervals;

--Creating new icustayid without duplication
DROP TABLE IF EXISTS mimic3_sepsis_final.SOFA_PERIOD_ADULT;
CREATE TABLE mimic3_sepsis_final.SOFA_PERIOD_ADULT
SELECT ICUSTAYID,STARTTIME,ENDTIME,MAX_SOFA,NUMBER, ICUSTAYID*10+NUMBER AS NEW_ICUSTAYID
FROM mimic3_sepsis_final.SOFA_PERIOD
JOIN mimic3_sepsis_final.DEMOG ON DEMOG.icustay_id = SOFA_PERIOD.ICUSTAYID
WHERE DEMOG.age > 6574
order by icustay_id