
1.1 CREATE TABLE
CREATE TABLE employees_copy (
   employee_id   NUMBER PRIMARY KEY,
   last_name     VARCHAR2(50),
   salary        NUMBER(8,2),
   hire_date     DATE
);

Create table from another table
CREATE TABLE emp_copy AS
SELECT * FROM employees;

1.2 ALTER TABLE
Add column
ALTER TABLE emp_copy
ADD bonus NUMBER(8,2);

Add multiple columns
ALTER TABLE emp_copy
ADD (
   bonus NUMBER,
   grade VARCHAR2(10)
);

Modify column datatype / size
ALTER TABLE emp_copy
MODIFY bonus NUMBER(10,2);

Rename column
ALTER TABLE emp_copy
RENAME COLUMN bonus TO loyalty_bonus;

Drop column
ALTER TABLE emp_copy
DROP COLUMN loyalty_bonus;

1.3 Constraints
Add primary key
ALTER TABLE emp_copy
ADD CONSTRAINT emp_copy_pk PRIMARY KEY (employee_id);

Add foreign key
ALTER TABLE emp_copy
ADD CONSTRAINT emp_copy_fk
FOREIGN KEY (department_id)
REFERENCES departments(department_id);

Drop constraint
ALTER TABLE emp_copy
DROP CONSTRAINT emp_copy_pk;

1.4 DROP TABLE
DROP TABLE emp_copy;


⚠ Deletes table structure and data

1.5 TRUNCATE TABLE
TRUNCATE TABLE emp_copy;


✔ Deletes all rows
❌ Cannot be rolled back

🔷 2. DML (Data Manipulation Language)

👉 Manipulates table data

2.1 INSERT
Insert one row
INSERT INTO emp_copy (employee_id, last_name, salary, hire_date)
VALUES (100, 'Smith', 5000, SYSDATE);

Insert multiple rows
INSERT ALL
   INTO emp_copy VALUES (101, 'Jones', 4000, SYSDATE)
   INTO emp_copy VALUES (102, 'Brown', 4500, SYSDATE)
SELECT * FROM dual;

Insert from another table
INSERT INTO emp_copy
SELECT * FROM employees;

2.2 UPDATE
Update all rows
UPDATE emp_copy
SET salary = salary * 1.10;

Update with condition
UPDATE emp_copy
SET salary = salary * 1.10
WHERE department_id = 50;

Update using subquery
UPDATE emp_copy
SET salary = (
   SELECT AVG(salary)
   FROM employees
)
WHERE employee_id = 100;

2.3 DELETE
Delete specific rows
DELETE FROM emp_copy
WHERE salary < 3000;

Delete all rows
DELETE FROM emp_copy;


⚠ Slower than TRUNCATE

2.4 MERGE (Insert or Update)
MERGE INTO emp_copy e
USING employees s
ON (e.employee_id = s.employee_id)
WHEN MATCHED THEN
   UPDATE SET e.salary = s.salary
WHEN NOT MATCHED THEN
   INSERT VALUES (s.employee_id, s.last_name, s.salary, s.hire_date);

3. TCL (Transaction Control Language)
COMMIT
COMMIT;

ROLLBACK
ROLLBACK;

SAVEPOINT
SAVEPOINT sp1;
ROLLBACK TO sp1;

4. SELECT (Querying Data)
Basic SELECT
SELECT * FROM emp_copy;

WHERE
SELECT * FROM emp_copy
WHERE salary > 4000;

ORDER BY
SELECT * FROM emp_copy
ORDER BY salary DESC;

GROUP BY & HAVING
SELECT department_id, COUNT(*)
FROM emp_copy
GROUP BY department_id
HAVING COUNT(*) > 5;

Subqueries
SELECT *
FROM emp_copy
WHERE salary = (SELECT MAX(salary) FROM emp_copy);