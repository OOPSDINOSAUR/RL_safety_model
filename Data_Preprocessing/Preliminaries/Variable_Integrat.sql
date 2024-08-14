USE mimic3_sepsis_cleaned;
DROP TABLE IF EXISTS mimic3_sepsis_final.LABS;
CREATE TABLE mimic3_sepsis_final.LABS AS
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM POTASSIUM_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM SODIUM_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM CHLORIDE_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM GLUCOSE_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM BUN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM CREATININE_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM MAGNESIUM_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM CALCIUM_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM CO2_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM SGOT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM SGPT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM BILIRUBIN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM TOTAL_PROTEIN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM ALBUMIN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM TROPONIN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM CRP_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM HEMOGLOBIN_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM HEMATOCRIT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM RBC_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM WBC_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PLATELET_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PTT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM ACT_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM INR_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PH_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PAO2_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM PACO2_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM BASE_EXCESS_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM LACTATE_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM BICARBONATE_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM ETCO2_CELE_CLEANED
UNION ALL
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM SVO2_CELE_CLEANED

DELETE FROM LABS
WHERE VALUE LIKE '%>%' or VALUE LIKE '%<%' or VALUE LIKE '''%' or VALUE LIKE '，%' or VALUE LIKE '=%' or VALUE REGEXP '^[A-Za-z]' or VALUE LIKE '`%' or VALUE LIKE '%/%' or VALUE LIKE '%,%'

#DMOG
CREATE TABLE mimic3_sepsis_final.DMOG
SELECT
    ad.subject_id, ad.hadm_id, i.icustay_id,
    UNIX_TIMESTAMP(ad.admittime) AS admittime,
    UNIX_TIMESTAMP(ad.dischtime) AS dischtime,
    ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
    CASE
        WHEN i.first_careunit='NICU' THEN 5
        WHEN i.first_careunit='SICU' THEN 2
        WHEN i.first_careunit='CSRU' THEN 4
        WHEN i.first_careunit='CCU' THEN 6
        WHEN i.first_careunit='MICU' THEN 1
        WHEN i.first_careunit='TSICU' THEN 3
    END AS unit,
    UNIX_TIMESTAMP(i.intime) AS intime,
    UNIX_TIMESTAMP(i.outtime) AS outtime,
    i.los,
    TIMESTAMPDIFF(DAY, p.dob, i.intime) AS age,
    UNIX_TIMESTAMP(p.dob) AS dob,
    UNIX_TIMESTAMP(p.dod) AS dod,
    p.expire_flag,
    CASE
        WHEN p.gender='M' THEN 1
        WHEN p.gender='F' THEN 2
    END AS gender,
    CASE WHEN TIMESTAMPDIFF(SECOND, p.dod, ad.dischtime) <= 24*3600 THEN 1 ELSE 0 END AS morta_hosp,
    CASE WHEN TIMESTAMPDIFF(SECOND, p.dod, i.intime) <= 90*24*3600 THEN 1 ELSE 0 END AS morta_90,
    congestive_heart_failure + cardiac_arrhythmias + valvular_disease + pulmonary_circulation + peripheral_vascular + hypertension + paralysis + other_neurological + chronic_pulmonary + diabetes_uncomplicated + diabetes_complicated + hypothyroidism + renal_failure + liver_disease + peptic_ulcer + aids + lymphoma + metastatic_cancer + solid_tumor + rheumatoid_arthritis + coagulopathy + obesity + weight_loss + fluid_electrolyte + blood_loss_anemia + deficiency_anemias + alcohol_abuse + drug_abuse + psychoses + depression AS elixhauser
FROM mimic3.ADMISSIONS ad, mimic3.ICUSTAYS i, mimic3.PATIENTS p, mimic3.elixhauser_quan elix
where ad.hadm_id=i.hadm_id and p.subject_id=i.subject_id and elix.hadm_id=ad.hadm_id
order by subject_id asc, intime asc

#PREADM_UO
DROP TABLE IF EXISTS mimic3_sepsis_final.PREADM_UO;
CREATE TABLE mimic3_sepsis_final.PREADM_UO
select distinct oe.icustay_id, UNIX_TIMESTAMP(oe.charttime) AS charttime, oe.itemid, oe.value,
                TIMESTAMPDIFF(MINUTE , oe.charttime, ic.intime) as datediff_minutes
from mimic3.OUTPUTEVENTS oe, mimic3.ICUSTAYS ic
where oe.icustay_id=ic.icustay_id
and itemid in (40060,226633)
order by icustay_id, charttime, itemid

#PREADM_FLUID
CREATE TABLE mimic3_sepsis_final.PREADM_FLUID AS
with mv as
(
select ie.icustay_id, sum(ie.amount) as sum
from mimic3.INPUTEVENTS_MV ie, mimic3.D_ITEMS ci
where ie.itemid=ci.itemid and ie.itemid in
(30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
227071,227072)
group by icustay_id
),

cv as
(
select ie.icustay_id, sum(ie.amount) as sum
from mimic3.INPUTEVENTS_CV ie, mimic3.D_ITEMS ci
where ie.itemid=ci.itemid and ie.itemid in
(30054,30055,30101,30102,30103,30104,30105,30108,226361,226363,226364,
226365,226367,226368,226369,226370,226371,226372,226375,226376,227070,
227071,227072)
group by icustay_id
)

select pt.icustay_id,
case when mv.sum is not null then mv.sum
when cv.sum is not null then cv.sum
else null end as inputpreadm
from mimic3.ICUSTAYS pt
left outer join mv
on mv.icustay_id=pt.icustay_id
left outer join cv
on cv.icustay_id=pt.icustay_id
order by icustay_id

#MECHVENT
CREATE TABLE mimic3_sepsis_final.MECHVENT AS

select ce.ICUSTAY_ID, UNIX_TIMESTAMP(ce.CHARTTIME) AS CHARTTIME,
    max(
    case
      when itemid is null or value is null then 0 -- can't have null values
      when itemid = 720 and value != 'Other/Remarks' THEN 1  -- VentTypeRecorded
      when itemid = 223848 and value != 'Other' THEN 1
      when itemid = 223849 then 1 -- ventilator mode
      when itemid = 467 and value = 'Ventilator' THEN 1 -- O2 delivery device == ventilator
      when itemid in
        (
        445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
        , 639,654,681,682,683,684,2400,2408,2420,2534,3050,3083,224685,224684,224686 -- tidal volume
        , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
        , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
        , 543,224696 -- Plateau Pressure
        , 5865,5866,224707,224709,224705,224706 -- APRV pressure
        , 60,437,505,506,686,1096,1227,5964,6349,6350,6489,6601,6924,6943,7802,7803,220339,224699,224700 -- PEEP
        , 3459 -- high pressure relief
        , 501,502,503,224702 -- PCV
        , 223,667,668,669,670,671,672 -- TCPCV
        , 224701 -- PSVlevel
        )
        THEN 1
      else 0
    end
    ) as MechVent,
    max(
	case when itemid is null or value is null then 0
	 when itemid = 640 and value = 'Extubated' then 1
	 when itemid = 640 and value = 'Self Extubation' then 1
	 else 0
	 end
	) as Extubated,
	max(
	 case when itemid is null or value is null then 0
	 when itemid = 640 and value = 'Self Extubation' then 1
	 else 0
	 end
	) as SelfExtubated
  FROM mimic3.CHARTEVENTS ce
  WHERE ce.value is not null AND (ce.error IS NULL or ce.error = 0) AND ce.itemid in
  (
	 640 -- extubated
	 , 720 -- vent type
	 , 467 -- O2 delivery device
	 , 445, 448, 449, 450, 1340, 1486, 1600, 224687 -- minute volume
	 , 639,654,681,682,683,684,2400,2408,2420,2534,3050,3083,224685,224684,224686 -- tidal volume
	 , 218,436,535,444,459,224697,224695,224696,224746,224747 -- High/Low/Peak/Mean/Neg insp force ("RespPressure")
	 , 221,1,1211,1655,2000,226873,224738,224419,224750,227187 -- Insp pressure
	 , 543,224696 -- PlateauPressure
	 , 5865,5866,224707,224709,224705,224706 -- APRV pressure
	 , 60,437,505,506,686,1096,1227,5964,6349,6350,6489,6601,6924,6943,7802,7803,220339,224699,224700 -- PEEP
	 , 3459 -- high pressure relief
	 , 501,502,503,224702 -- PCV
	 , 223,667,668,669,670,671,672 -- TCPCV
	 , 157,158,1852,3398,3399,3400,3401,3402,3403,3404,8382,227809,227810 -- ETT
	 , 224701 -- PSVlevel
  )
  group by icustay_id, charttime


ALTER TABLE mimic3_sepsis_cleaned.ALL_VASO_MV
ADD COLUMN rate_std NUMERIC(10, 3);

UPDATE mimic3_sepsis_cleaned.ALL_VASO_MV
SET rate_std =
    CASE
        when itemid in (30120,221906,30047) and rateuom='mcg/kg/min' then round(rate,3)  -- norad
        when itemid in (30120,221906,30047) and rateuom='mcg/min' then round(rate/80,3)  -- norad
        when itemid in (30119,221289,30309,30044) and rateuom='mcg/kg/min' then round(rate ,3) -- epi
        when itemid in (30119,221289,30309,30044) and rateuom='mcg/min' then round(rate/80 ,3) -- epi
        when itemid in (30051,222315) and rate > 0.2 then round(rate*5/60 ,3) -- vasopressin, in U/h
        when itemid in (30051,222315) and rateuom='units/min' then round(rate*5 ,3) -- vasopressin
        when itemid in (30051,222315) and rateuom='units/hour' then round(rate*5/60 ,3) -- vasopressin
        when itemid in (30128,221749,30127) and rateuom='mcg/kg/min' then round(rate*0.45 ,3) -- phenyl
        when itemid in (30128,221749,30127) and rateuom='mcg/min' then round(rate*0.45 / 80 ,3) -- phenyl
        when itemid in (221662,30043,30307) and rateuom='mcg/kg/min' then round(rate*0.01 ,3)  -- dopa
        when itemid in (221662,30043,30307) and rateuom='mcg/min' then round(rate*0.01/80 ,3) else null
    END
where itemid in (30128,30120,30051,221749,221906,30119,30047,30127,221289,222315,221662,30043,30307,30309,30044)

ALTER TABLE mimic3_sepsis_cleaned.ALL_VASO_CV
ADD COLUMN rate_std NUMERIC(10, 3);

UPDATE mimic3_sepsis_cleaned.ALL_VASO_CV
SET rate_std =
    CASE
        when itemid in (30120,221906,30047) and rateuom='mcgkgmin' then round(rate,3)  -- norad
        when itemid in (30120,221906,30047) and rateuom='mcgmin' then round(rate/80,3)  -- norad
        when itemid in (30119,221289,30309,30044) and rateuom='mcgkgmin' then round(rate ,3) -- epi
        when itemid in (30119,221289,30309,30044) and rateuom='mcgmin' then round(rate/80 ,3) -- epi
        when itemid in (30051,222315) and rate > 0.2 then round(rate*5/60 ,3) -- vasopressin, in U/h
        when itemid in (30051,222315) and rateuom='Umin' and rate < 0.2 then round(rate*5 ,3) -- vasopressin
        when itemid in (30051,222315) and rateuom='Uhr' then round(rate*5/60 ,3) -- vasopressin
        when itemid in (30128,221749,30127) and rateuom='mcgkgmin' then round(rate*0.45 ,3) -- phenyl
        when itemid in (30128,221749,30127) and rateuom='mcgmin' then round(rate*0.45 / 80 ,3) -- phenyl
        when itemid in (221662,30043,30307) and rateuom='mcgkgmin' then round(rate*0.01 ,3)  -- dopa
        when itemid in (221662,30043,30307) and rateuom='mcgmin' then round(rate*0.01/80 ,3) else null
    END
where itemid in (30128,30120,30051,221749,221906,30119,30047,30127,221289,222315,221662,30043,30307,30309,30044)

CREATE TABLE mimic3_sepsis_final.VASO_CV
SELECT ICUSTAY_ID, ITEMID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, rate_std
FROM mimic3_sepsis_cleaned.ALL_VASO_CV;

CREATE TABLE mimic3_sepsis_final.VASO_MV
SELECT ICUSTAY_ID, ITEMID, UNIX_TIMESTAMP(STARTTIME) AS STARTTIME, UNIX_TIMESTAMP(ENDTIME) AS ENDTIME,rate_std
FROM mimic3_sepsis_cleaned.ALL_VASO_MV

DROP TABLE IF EXISTS mimic3_sepsis_final.CE;
CREATE TABLE mimic3_sepsis_final.CE
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.HEIGHT_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.WEIGHT_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.GCS_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.RASS_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.HEART_RATE_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.BP_DIA_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.BP_MEAN_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.BP_SYS_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.RESP_RATE_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.SPO2_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.TEMPERATURE_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.CVP_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAP_MEAN_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAP_SYS_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAP_DIA_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.CARDIAC_INDEX_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.SVR_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.O2_DELI_DEV_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.FIO2_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.O2_FLOW_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PEEP_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_new.Tidal_Volume
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.MINUTE_VOLUME_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAW_MEAN_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAW_PEAK_CLEANED
UNION
SELECT ICUSTAY_ID, UNIX_TIMESTAMP(CHARTTIME) AS CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_cleaned.PAW_PLATEAU_CLEANED

DELETE FROM CE
WHERE  VALUE LIKE '%/%'

CREATE TABLE mimic3_sepsis_final.CE_0_10000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 200000 AND ICUSTAY_ID < 210000

CREATE TABLE mimic3_sepsis_final.CE_10000_20000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 210000 AND ICUSTAY_ID < 220000

CREATE TABLE mimic3_sepsis_final.CE_20000_30000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 220000 AND ICUSTAY_ID < 230000

CREATE TABLE mimic3_sepsis_final.CE_30000_40000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 230000 AND ICUSTAY_ID < 240000

CREATE TABLE mimic3_sepsis_final.CE_40000_50000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 240000 AND ICUSTAY_ID < 250000

CREATE TABLE mimic3_sepsis_final.CE_50000_60000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 250000 AND ICUSTAY_ID < 260000

CREATE TABLE mimic3_sepsis_final.CE_60000_70000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 260000 AND ICUSTAY_ID < 270000

CREATE TABLE mimic3_sepsis_final.CE_70000_80000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 270000 AND ICUSTAY_ID < 280000

CREATE TABLE mimic3_sepsis_final.CE_80000_90000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 280000 AND ICUSTAY_ID < 290000

CREATE TABLE mimic3_sepsis_final.CE_90000_100000
SELECT ICUSTAY_ID, CHARTTIME, ITEMID, VALUE
FROM mimic3_sepsis_final.CE
WHERE ICUSTAY_ID >= 290000 AND ICUSTAY_ID <300000


CREATE TABLE mimic3_sepsis_new.ALL_FLUID_CV AS
SELECT * FROM mimic3_sepsis_new.ALBUMIN_CV
WHERE 1 = 0
UNION ALL
SELECT * FROM mimic3_sepsis_new.CRYO_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.DEXTRAN_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.DEXTROSE_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.PLASMA_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.HEPARIN_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.KCL_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.LACT_RINGER_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.MANNITOL_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.NACL_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.PLASMALYTE_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.PLATELET_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.RBC_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.NAHCO3_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.SSC_CV
UNION ALL
SELECT * FROM mimic3_sepsis_new.ALBUMIN_CV;

CREATE TABLE mimic3_sepsis_final.FLUID_CV AS (
with t2 AS (select icustay_id, UNIX_TIMESTAMP(charttime) as charttime, itemid, amount,
        case when itemid in (30176,30315,30212,41459,30316) then amount * 0.25
             when itemid in(45325,46027,30018,30160,30168,30352,43354,44053,44440,30296,30190,30210,40850,40865,41380,
                            41467,41490,41695,41913,42548,42844,43663,44491,44983,226401,225158) then amount * 0.9
             when itemid in(30143,225161) then amount * 3
             when itemid in(30161) then amount * 0.3
             when itemid in(30020,30015,225823,30321,30186,30211,30353,42742,42244,225159,41526,43232,43406,43588,44439) then amount * 0.45
             when itemid in(228341) then amount * 8
             when itemid in(30143,225161) then amount * 3
             when itemid in(227531,42172,220991) then amount * 2.75
             when itemid in(220988) then amount * 2.75 * 0.25
             when itemid in(220989) then amount * 2.75 * 0.5
             when itemid in(220990) then amount * 2.75 * 0.75
             when itemid in(220992) then amount * 2.75 * 1.25
             when itemid in(30009,220862,43237,43353) then amount * 5
             when itemid in(42832,44203) then amount * 2.5
             when itemid in(30030,220995,227533,41983,44602) then amount * 6.66
             when itemid in(221211) then amount * 1.11 else amount end as tev -- total equivalent volume
    from mimic3_sepsis_new.ALL_FLUID_CV
    where amount is not null AND AMOUNT REGEXP '^(?:[0-9]+)?(\.[0-9]+)?$'
    order by icustay_id, charttime, itemid)
select icustay_id,charttime,itemid,round(amount, 3) as amount,round(tev, 3) as tev -- total equivalent volume
                                              from t2
                                              order by icustay_id, charttime, itemid)


CREATE TABLE mimic3_sepsis_final.ALL_FLUID_MV AS
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.ALBUMIN_MV
WHERE 1 = 0
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.CRYO_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.DEXTRAN_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.DEXTROSE_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.PLASMA_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.HEPARIN_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.KCL_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.LACT_RINGER_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.MANNITOL_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.NACL_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.PLATELET_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.RBC_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.NAHCO3_MV
UNION ALL
SELECT icustay_id, STARTTIME,  ENDTIME,itemid, amount,rate FROM mimic3_sepsis_new.ALBUMIN_MV;

CREATE TABLE mimic3_sepsis_final.FLUID_MV AS (
with t2 AS (select icustay_id, UNIX_TIMESTAMP(STARTTIME) as STARTTIME, UNIX_TIMESTAMP(ENDTIME) as ENDTIME,itemid, amount,rate,
        case when itemid in (30176,30315,30212,41459,30316) then amount * 0.25
             when itemid in(45325,46027,30018,30160,30168,30352,43354,44053,44440,30296,30190,30210,40850,40865,41380,
                            41467,41490,41695,41913,42548,42844,43663,44491,44983,226401,225158) then amount * 0.9
             when itemid in(30143,225161) then amount * 3
             when itemid in(30161) then amount * 0.3
             when itemid in(30020,30015,225823,30321,30186,30211,30353,42742,42244,225159,41526,43232,43406,43588,44439) then amount * 0.45
             when itemid in(228341) then amount * 8
             when itemid in(30143,225161) then amount * 3
             when itemid in(227531,42172,220991) then amount * 2.75
             when itemid in(220988) then amount * 2.75 * 0.25
             when itemid in(220989) then amount * 2.75 * 0.5
             when itemid in(220990) then amount * 2.75 * 0.75
             when itemid in(220992) then amount * 2.75 * 1.25
             when itemid in(30009,220862,43237,43353) then amount * 5
             when itemid in(42832,44203) then amount * 2.5
             when itemid in(30030,220995,227533,41983,44602) then amount * 6.66
             when itemid in(221211) then amount * 1.11 else amount end as tev -- total equivalent volume
    from mimic3_sepsis_new.ALL_FLUID_MV
    where amount is not null AND AMOUNT REGEXP '^(?:[0-9]+)?(\.[0-9]+)?$'
    order by icustay_id, STARTTIME, itemid)
select icustay_id,STARTTIME,ENDTIME,itemid,round(amount, 3) as amount,round(rate, 3) as rate,round(tev, 3) as tev -- total equivalent volume
                                              from t2
                                              order by icustay_id, STARTTIME, itemid);

