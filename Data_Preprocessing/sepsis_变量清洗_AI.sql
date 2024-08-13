#Culture
USE mimic3_sepsis_cleaned;
CREATE TABLE CULTURE_CE AS
select subject_id, hadm_id, icustay_id, charttime, itemid, error
FROM mimic3.CHARTEVENTS
WHERE (ERROR IS NULL OR ERROR = 0)
and itemid in
(938,941,942,2929,3333,4855,6035,6043,225722,225723,225724,
225725,225726,225727,225728,225729,225730,225731,225732,
225733,225734,225735,225736,225768,226131,227726)
order by subject_id, hadm_id, charttime;

CREATE TABLE CULTURE_ME AS
-- Culture, in MICROBIOLOGYEVENTS
select subject_id, hadm_id,
(case when CHARTTIME IS NULL then CHARTDATE ELSE CHARTTIME end) AS charttime,
spec_itemid as itemid
FROM mimic3.MICROBIOLOGYEVENTS
where spec_itemid in
(70006,70011,70012,70013,70014,70016,70024,70037,70041,70055,
70057,70060,70063,70075,70083,80220)
order by subject_id, hadm_id, charttime;

CREATE TABLE CULTURE_PE AS
-- Culture, in PROCEDUREEVENTS_MV
select subject_id, hadm_id, icustay_id, endtime as charttime, itemid
FROM mimic3.PROCEDUREEVENTS_MV
WHERE STATUSDESCRIPTION != 'Rewritten'
and itemid in
(225401,225437,225444,225451,225454,225814,225816,225817,225818)
order by subject_id, hadm_id, charttime;

-- ALL MICROBIOLOGYEVENTS
CREATE TABLE mimic3_sepsis_new.ME AS
select subject_id, hadm_id,
(case when CHARTTIME IS NULL then CHARTDATE ELSE CHARTTIME end) AS charttime
FROM mimic3.MICROBIOLOGYEVENTS;

#Antibiotics administration:
CREATE TABLE mimic3_sepsis_cleaned.ANTIBIOTICS AS
select subject_id, hadm_id, icustay_id, startdate, enddate
from mimic3.PRESCRIPTIONS
where gsn in ('002542','002543','007371','008873','008877',
'008879','008880','008935','008941','008942','008943','008944',
'008983','008984','008990','008991','008992','008995','008996',
'008998','009043','009046','009065','009066','009136','009137',
'009162','009164','009165','009171','009182','009189','009213',
'009214','009218','009219','009221','009226','009227','009235',
'009242','009263','009273','009284','009298','009299','009310',
'009322','009323','009326','009327','009339','009346','009351',
'009354','009362','009394','009395','009396','009509','009510',
'009511','009544','009585','009591','009592','009630','013023',
'013645','013723','013724','013725','014182','014500','015979',
'016368','016373','016408','016931','016932','016949','018636',
'018637','018766','019283','021187','021205','021735','021871',
'023372','023989','024095','024194','024668','025080','026721',
'027252','027465','027470','029325','029927','029928','037042',
'039551','039806','040819','041798','043350','043879','044143',
'045131','045132','046771','047797','048077','048262','048266',
'048292','049835','050442','050443','051932','052050','060365',
'066295','067471')
order by subject_id, hadm_id, icustay_id;

#DEMOGRAPHICS
CREATE TABLE mimic3_sepsis_cleaned.DEMOGRAPHICS AS
SELECT
	ad.subject_id, ad.hadm_id, i.icustay_id, ad.admittime, ad.dischtime,
	ROW_NUMBER() OVER (PARTITION BY ad.subject_id ORDER BY i.intime ASC) AS adm_order,
	CASE
		WHEN i.first_careunit='NICU' THEN 5
		WHEN i.first_careunit='SICU' THEN 2
		WHEN i.first_careunit='CSRU' THEN 4
		WHEN i.first_careunit='CCU' THEN 6
		WHEN i.first_careunit='MICU' THEN 1
		WHEN i.first_careunit='TSICU' THEN 3
	END AS unit,
	i.intime, i.outtime, i.los,
	TIMESTAMPDIFF(YEAR, p.dob, i.intime) AS age,
	p.dob, p.dod,
	p.expire_flag,
	CASE
		WHEN p.gender='M' THEN 1
		WHEN p.gender='F' THEN 2
	END AS gender,
	CASE
		WHEN TIMESTAMPDIFF(SECOND, p.dod, ad.dischtime) <= 24*3600 THEN 1
		ELSE 0
	END AS morta_hosp,
	CASE
		WHEN TIMESTAMPDIFF(SECOND, p.dod, i.intime) <= 90*24*3600 THEN 1
		ELSE 0
	END AS morta_90,
	congestive_heart_failure + cardiac_arrhythmias +
	valvular_disease + pulmonary_circulation + peripheral_vascular +
	hypertension + paralysis + other_neurological + chronic_pulmonary +
	diabetes_uncomplicated + diabetes_complicated + hypothyroidism +
	renal_failure + liver_disease + peptic_ulcer + aids + lymphoma +
	metastatic_cancer + solid_tumor + rheumatoid_arthritis + coagulopathy +
	obesity + weight_loss + fluid_electrolyte + blood_loss_anemia +
	deficiency_anemias + alcohol_abuse + drug_abuse + psychoses +
	depression AS elixhauser
FROM
	mimic3.ADMISSIONS ad,
	mimic3.ICUSTAYS i,
	mimic3.PATIENTS p,
	mimic3.elixhauser_quan elix
where
	ad.hadm_id=i.hadm_id and
	p.subject_id=i.subject_id and
	elix.hadm_id=ad.hadm_id
order by
	subject_id asc, intime asc

#Real-time UO
CREATE TABLE mimic3_sepsis_cleaned.UO AS
select icustay_id, charttime, itemid, value, valueuom
from mimic3.OUTPUTEVENTS
where icustay_id is not null and value is not null
and itemid in
(40055 ,43175 ,40069, 40094 ,40715 ,40473 ,40085,
40057, 40056 ,40405 ,40428, 40096, 40651,226559,
226560 ,227510 ,226561 ,227489 ,226584, 226563,
226564 ,226565 ,226557 ,226558)
order by icustay_id, charttime, itemid;

#preadmission-uo
CREATE TABLE mimic3_sepsis_cleaned.PREADM_UO AS
select distinct oe.icustay_id, oe.charttime, oe.itemid, oe.value, oe.valueuom,
TIMESTAMPDIFF(MINUTE , oe.charttime, ic.intime) as datediff_minutes
from mimic3.OUTPUTEVENTS oe, mimic3.ICUSTAYS ic
where oe.icustay_id=ic.icustay_id
and itemid in (40060,226633)
order by icustay_id, charttime, itemid

#preadm_fluid
CREATE TABLE mimic3_sepsis_cleaned.PREADM_FLUID AS
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

#mech_vent
CREATE TABLE mimic3_sepsis_cleaned.MECHVENT AS

select ce.ICUSTAY_ID, ce.CHARTTIME,
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



