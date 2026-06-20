
1.
alter table employees 
modify last_name varchar2(50)

2.
create table employees2 as
select employee_id, first_name, last_name, salary, department_id from employees;

alter table employees2 rename column(
   employee_id as id,
   departmend_id as dept_id
)

3.
drop table employees2;

4.
alter table employees2 rename emp;

6.
alter table emp drop column first_name

9.
alter table emp add constraint my_emp_id_pk primary key employee_id

10.
alter table departments add constraint my_dept_id_pk primary key department_id

11.
alter table emp add constraint my_emp_dept_id_fk foreign key department_id
references departments.department_id;

12.
alter table emp drop constraint my_emp_id_pk, my_dept_id_pk, my_emp_dept_id_fk

14.
alter table emp add comission number;

alter table emp add constraint gt_zero 
CHECK (commission >= 0)

15.
create table dept
as select department_id, department_name from departments;

16.
update emp set (employee_id, department_id, salary, commission_pct)
as select employee_id, department_id, salary, commission_pct from employees
where department_id = 80;

17.
alter table emp rename column employee_id to PR 
where employee_id = 70;

18.
alter table emp add total_sal number;

alter table emp add constraint my_constraint total_sal cheksum (
    select count(salary) from emp
    group by first_name
    )