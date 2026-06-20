
1.

create sequence sec_pk
start with 0
increment by 1;

CREATE OR REPLACE FUNCTION valid_job_id (
   p_job_id jobs.job_id%TYPE
)
RETURN BOOLEAN
IS
   v_dummy NUMBER;
BEGIN
   SELECT 1
   INTO v_dummy
   FROM jobs
   WHERE job_id = p_job_id;

   RETURN TRUE;

EXCEPTION
   WHEN NO_DATA_FOUND THEN
      RETURN FALSE;
   WHEN OTHERS THEN
      RETURN FALSE;
END;
/

CREATE OR REPLACE PROCEDURE add_emp (
   p_name      IN emp_pnu.name%TYPE,
   p_job_id    IN emp_pnu.job_id%TYPE,
   p_salary    IN emp_pnu.salary%TYPE DEFAULT 3000,
   p_hire_date IN emp_pnu.hire_date%TYPE DEFAULT SYSDATE
)
IS
BEGIN
   IF valid_job_id(p_job_id) THEN

      INSERT INTO emp_pnu (
         emp_id,
         name,
         job_id,
         salary,
         hire_date
      )
      VALUES (
         emp_pnu_seq.NEXTVAL,
         p_name,
         p_job_id,
         p_salary,
         p_hire_date
      );

   ELSE
      RAISE_APPLICATION_ERROR(
         -20001,
         'Invalid job ID'
      );
   END IF;

EXCEPTION
   WHEN OTHERS THEN
      DBMS_OUTPUT.PUT_LINE('Error inserting employee: ' || SQLERRM);
END;
/


2.

CREATE OR REPLACE PROCEDURE dept_procedure (
   p_dept_id IN departments.department_id%TYPE
)
IS
   v_emp_count   NUMBER;
   v_salary_sum  NUMBER;
BEGIN
   SELECT COUNT(*),
          SUM(salary)
   INTO v_emp_count,
        v_salary_sum
   FROM employees
   WHERE department_id = p_dept_id;

   DBMS_OUTPUT.PUT_LINE(
      'Department ' || p_dept_id ||
      ': Employees = ' || v_emp_count ||
      ', Total Salary = ' || v_salary_sum
   );

EXCEPTION
   WHEN NO_DATA_FOUND THEN
      DBMS_OUTPUT.PUT_LINE('No employees found.');
END;
/

3.
DECLARE
   v_n NUMBER := &n;

   CURSOR c_emp IS
      SELECT last_name, salary, commission_pct
      FROM employees
      WHERE salary > v_n;
BEGIN
   FOR r IN c_emp LOOP
      IF r.commission_pct IS NOT NULL THEN
         DBMS_OUTPUT.PUT_LINE(
            r.last_name || ' ' || r.salary
         );
      END IF;
   END LOOP;
END;
/


4.

DECLARE
   e_no_employee EXCEPTION;
BEGIN
   DELETE FROM employees e
   WHERE e.commission_pct >
         ( SELECT (m.salary - e.salary) / 2
           FROM employees m
           WHERE m.employee_id = e.manager_id
         );

   IF SQL%ROWCOUNT = 0 THEN
      RAISE e_no_employee;
   END IF;

EXCEPTION
   WHEN e_no_employee THEN
      DBMS_OUTPUT.PUT_LINE('no employee with the specified commission');
END;
/


delete from employees
where commission_pct > (
    select m.salary - e.salary from employees e 
    join employees m on m.employee_id= e.manager_id
) /2






2.
DECLARE
CREATE PROCEDURE PROC_DEPT_CODE(
   V_DEPT_ID IN DEPARTMENTS.DEPARTMENT_ID%TYPE
)

V_EMP_COUNT NUMBER;
V_SALARY_SUM NUMBER;
BEGIN
   SELECT V_EMP_COUNT, V_SALARY_SUM
   INTO COUNT(e.LAST_NAME), sum(e.salary)
   FROM EMPLOYEES e
   JOIN DEPARTMENTS D ON D. DEPARTMENT_ID = E.employee_id
   where d.department_id = V_DEPT_ID
   GROUP BY V_DEPT_ID

   DBMS.OUTPUT_PUT_LINE(
       'Department ' || p_dept_id ||
      ': Employees = ' || v_emp_count ||
      ', Total Salary = ' || v_salary_sum)


   exception 
      when no_data_found THEN
       DBMS.OUTPUT_PUT_LINE('No data found');
END


5.
create trigger my_trigger
    before insert employees.last_name
       for each ROW 
       when (salary < 1000)
BEGIN
   RAISE_APPLICATION_ERROR(
      -20001,
      'Cannot insert salary less than 1000'
   );
END;
/


4.
DECLARE
   v_emp_id employees.employee_id%TYPE;
BEGIN
   delete from employees(
      select v_emp_id
      into employee_id
      from employees
      where commission_pct (
       select m.salary - e.salary from employees
       join employees m on m.manager_id = e.employee_id
   ) / 2)
   )
   
  IF NO_DATA_FOUND RAISE_APPLICATION_ERROR ('no employee with the specified commission'.)
END



delete from employees(
   select employee_id from employees
   where commission_pct (
       select m.salary - e.salary from employees
       join employees m on m.manager_id = e.employee_id
   ) / 2)