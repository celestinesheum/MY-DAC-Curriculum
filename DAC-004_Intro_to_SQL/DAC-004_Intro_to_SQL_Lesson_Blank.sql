-- Refresher on how to perform basic query and how the database works:

-- SELECT Clause: everything = *

-- Select department table, the employee table and vendor table. Let's explore the database a little!

SELECT *
FROM humanresources.employee

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- SELECT some columns:

-- Select only name, start time and end time.



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- SELECT DISTINCT values: Unique column value

-- Distinct group names from department and businessentityid from jobcandidate



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- From different schemas: sales



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- LIMIT: As the name suggest it limits the number of *rows* shown at the end result

-- Limit the table productvendor to 10 rows and purchaseorderdetail to 100 rows



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- SELECT MDAS: Multiplcation/division/addition/subtraction

-- From the customer table Multiplcation/division/addition/subtraction the store_id



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--Q1: SELECT the DISTINCT title, last name, middlename and first_name of each person from the person schema. Return only 231 rows.
--A1;



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- WHERE clause: = 
-- gender is male


-- Only Research and Development


-- When dealing with NULL values


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- WHERE clause: Arithmetic filter

-- From customer table, territoryid = 4


-- From person table, emailpromotion <> 0


-- From employee table, vacationhours >= 99


-- From employee table, sickleavehours <= 20


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--WHERE clause: OR clause

-- From employee table, select either Design Engineer or Tool Designer


-- From product, select either Black or Silver


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- WHERE clause: AND clause

-- From Vendor, preferredvendorstatus and activeflag must be TRUE


-- From employee, gender must be Male and maritalstatus must be single


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--WHERE clause: Combined OR & AND clause

-- From the employee table pick either, marital status as single and gender male or marital status as married and gender female.


-- Example of poor formatting and logic.
-- From the salesperson table select territory_id either 4 or 6 and salesquota either 250000 or 300000

--


--


--Note: AND takes higher priority than OR

-- Reformatted version:
-- The importance of having good SQL formatting when writing your SQL code.



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--WHERE clause: IN clause
--Q: Find all the employees whose birthdate fall on these dates.

-- '1977-06-06'
-- '1984-04-30'
-- '1985-05-04'



-- Find all the middle names that contains either A or B or C.



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- WHERE clause: LIKE clause
-- The placement of the wildcard, %, affects what is getting filtered out.

-- From the person table, select all the firstname starting with a 'J'
-- Works very similar to excel find function
SELECT*
FROM person.person
WHERE firstname LIKE 'J%';
-- Find J

-- Only works for string!

-- But what if you know the number of letters in the firstname?

SELECT *
FROM person.person
WHERE firstname LIKE 'J___';

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- What if we want firstnames that contains the letter a inside?
SELECT*
FROM 

-- not tallying

-- We have two varying results, we can use things like UPPER() and LOWER() clause
SELECT* 
FROM person.person
WHERE UPPER(firstname) LIKE '%A%';

SELECT* 
FROM person.person
WHERE LOWER(firstname) LIKE '%a%';
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- WHERE clause: NOT clause

SELECT* 
FROM person.person
WHERE lastname LIKE '%A%';

SELECT* 
FROM person.person
WHERE UPPER(lastname) NOT LIKE '%a%';
-- From the person table, lastname should not contain A in it.
SELECT*
FROM person.person
WHERE lastname NOT LIKE '%A%';


-- From the employee table, choose those that do not fall into this date range:
--'1977-06-06', '1984-04-30','1985-05-04'

SELECT* 
FROM humanresources.employee
WHERE birthdate NOT IN (
	'1977-06-06',
	'1984-04-30',
	'1985-05-04'
)


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- GROUP BY clause: For aggregate values
-- For us to use when we want to use aggregates.

-- From employee table, group by gender

SELECT 
	gender
FROM humanresources.employee
GROUP BY gender;

-- From employee table, group by maritalstatus

SELECT 
	maritalstatus
FROM humanresources.employee
GROUP BY maritalstatus;

-- We can also group more than one column
SELECT 
	gender,
	maritalstatus,
	jobtitle
FROM humanresources.employee
GROUP BY gender,
	maritalstatus,
	jobtitle



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- All the AGGREGATES!
SELECT
	--gender,
	--COUNT(gender) AS Headcount --270ms
	--COUNT(*) AS Headcount --268ms
	gender, -- select gender column
	COUNT(*) AS Headcount, -- select all headcount
	COUNT(DISTINCT jobtitle) AS uniquejobtitles,
	SUM(vacationhours) AS total_vacation_hours,
	AVG(vacationhours) AS average_vacation_hours,
	CEILING(AVG(vacationhours)) AS ceiling_vacation_hours,
	FLOOR(AVG(vacationhours)) AS floor_vacation_hours,
	ROUND(AVG(vacationhours)) AS rounded_average,

	MAX(sickleavehours) AS max_sick_hours,
	MIN(sickleavehours) AS min_sick_hours
FROM humanresources.employee
GROUP BY gender;-- group by gender 



-- Q2: Analyse if the marital status of each gender affects the number of vacation hours one will take
-- A2:
SELECT
	gender,
	maritalstatus,
	AVG(vacationhours) AS average_vacation_hours
FROM humanresources.employee
GROUP BY 1, 2; -- 1 is gender, 2 is marital status
	

-- From employee table, ORDER BY hiredate, ASC and DESC


-- hiredate earliest
SELECT*
FROM humanresources.employee
ORDER BY hiredate ASC;

-- hiredate latest
SELECT*
FROM humanresources.employee
ORDER BY hiredate DESC;

-- Sort table using two or more values [EG: GENDER AND JOBTITLE]
SELECT
	jobtitle,
	gender
FROM humanresources.employee
ORDER BY jobtitle ASC, gender DESC;

-- Sorting by Average
SELECT
	jobtitle,
	AVG(vacationhours) AS avg_vacation_hours
FROM humanresources.employee
GROUP BY jobtitle
ORDER BY AVG(vacationhours) DESC

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- HAVING clause:
SELECT
	jobtitle,
	AVG(sickleavehours) AS avg_sick_leave_hours
FROM humanresources.employee
GROUP BY jobtitle
HAVING AVG(sickleavehours)>50;


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Q3: From the customer table, where customer has a personid and a storeid, find the territory that has higher than 40 customers
-- A3:
SELECT
	territoryid,
	COUNT(*) AS Number_of_customers
FROM sales.customer
WHERE personid IS NOT NULL
	AND storeid IS NOT NULL
GROUP BY territoryid
HAVING COUNT(*)>40;


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- OFFSET: Using the employee table find the other the other employees except the top 10 oldest employees.
SELECT *
FROM humanresources.employee
ORDER BY birthdate ASC;

SELECT *
FROM humanresources.employee
ORDER BY birthdate ASC
OFFSET 10;



-- Q4: From the salesperson table, where customer has a personid and a storeid, find the territory that has higher than 40 customers
-- A4:


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Best practise: When exploring a new table:
/*
	Why should we use the example mentioned below?
	1) We don't have to generate the entire table to understand what kind of information the table stores.
	2) Much faster using this compared to generating the entire multi-million row table
	3) So people don't think you are a noob
*/

SELECT*
FROM humanresources.employee
WHERE gender = 'M'
LIMIT 10;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- JOINS: INNER

-- Inner join to get product information along with its subcategory name and category name

SELECT *
FROM production.product;

SELECT *
FROM production.productsubcategory;

SELECT *
FROM production.productcategory;

SELECT
	product.productid,
	product.name AS product_name,
	productcategory.name AS categoryname,
	productsubcategory.name AS subcategoryname
FROM production.product AS product --left table
INNER JOIN production.productsubcategory AS productsubcategory --right table
		ON product.productsubcategoryid = productsubcategory.productsubcategoryid
INNER JOIN production.productcategory AS productcategory
		ON productsubcategory.productcategoryid = productcategory.productcategoryid;
-- INNER JOIN
-- Let's create a base table in the humanresources schema, where we are able to get each employee's department history and department name

-- Employee table

SELECT *
FROM humanresources.employee;

-- Unique table or?


-- Employee Department History table

SELECT *
FROM humanresources.employeedepartmenthistory;

-- Unique table or?


-- Department table

SELECT *
FROM humanresources.department;



-- Let's find all the employee, their respecitve departments and the time they served there. Bonus if you can find out the duration in days each employee spent
-- in each department! Duration in days cannot be NULL.



-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- JOINS: LEFT

-- List all products along with their total sales quantities, including products that have never been sold. 
-- For products that have not been sold, display the sales quantity as zero.
-- Sort by total orders descending

SELECT *
FROM production.product;

SELECT *
FROM sales.salesorderdetail;

SELECT
	product.productid,
	product.name AS productname,
	COALESCE(SUM(salesorderdetail.orderqty),0) AS totalsalesquantity
FROM production.product AS product
LEFT JOIN sales.salesorderdetail AS salesorderdetail
	ON product.productid = salesorderdetail.productid
GROUP BY
	product.productid,
	product.name
ORDER BY
	COALESCE(SUM(salesorderdetail.orderqty),0) DESC;

-- Q5: List all employees and their associated email addresses,  
-- display their full name and email address.
SELECT*
FROM humanresources.employee

SELECT*
FROM person.person

SELECT*
FROM person.emailaddress

SELECT
	CONCAT(person.firstname, ' ', person.middlename, ' ',person.lastname)AS fullname,
	emailaddress.emailaddress AS email
FROM humanresources.employee AS employee
LEFT JOIN person.person AS person 
	ON employee.businessentityid = person.businessentityid
LEFT JOIN person.emailaddress AS emailaddress
	ON employee.businessentityid = emailaddress.businessentityid;


-- Retrieve a list of all individual customers id, firstname along with the total number of orders they have placed 
-- and the total amount they have spent, removing customers who have never placed an order. 

SELECT *
FROM person.person;

SELECT *
FROM sales.customer;

SELECT *
FROM sales.salesorderheader;

SELECT
	customer.customerid,
	person.firstname,
	COUNT(salesorderid) AS purchases,
	ROUND(SUM(subtotal),2)AS cost
FROM sales.customer AS customer
LEFT JOIN person.person AS person 
	ON customer.personid = person.businessentityid
LEFT JOIN sales.salesorderheader
	ON customer.customerid = salesorderheader.customerid
GROUP BY
	customer.customerid,
	person.firstname
HAVING ROUND(SUM(subtotal),2) IS NOT NULL;
-- Q6: Can LEFT JOIN cause duplication? How?
-- A6: 
/*
It depends on the relationship that both the tables present for the left join share.
IF it is a one-to-one relationship, the chance of having duplicates is unlikely
HOweveer, if it is a one-to-many relationship, there could be a chance of duplicates to be present.
*/
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- JOINS: RIGHT
-- Write a query to retrieve all sales orders and their corresponding customers. If a sales order exists without an associated customer, 
-- include the sales order in the result.

-- SELECT 
--     salesorderheader.salesorderid AS salesorderid, 
--     salesorderheader.orderdate AS orderdate, 
--     customer.customerid AS customerid, 
--     customer.personid AS personid
-- FROM sales.salesorderheader AS salesorderheader;

SELECT
	salesorderheader.salesorderid AS salesorderid,
	salesorderheader.orderdate AS orderdate,
	customer.customerid AS customerid,
	customer.personid AS salesorderheader
FROM sales.salesorderheader AS salesorderheader
RIGHT JOIN sales.customer AS customer
	ON salesorderheader.customerid = customer.customerid;


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- JOINS: FULL OUTER JOIN

-- Write a query to find all employees and their corresponding sales orders. If an employee doesn’t have any sales orders, 
-- still include them in the result, and if there are sales orders without an associated employee, include those as well.

SELECT 
    employee.businessentityid AS employeeid,
    salesorderheader.salesorderid
FROM humanresources.employee AS employee
FULL OUTER JOIN sales.salesorderheader AS salesorderheader
	ON employee.businessentityid = salesorderheader.salespersonid;
	
SELECT*
FROM humanresources.employee
LIMIT 10;
		
-- Write a query to retrieve a list of all employees and customers, and if either side doesn't have a FirstName, 
-- use the available value from the other side. Use FULL OUTER JOIN and COALESCE.

SELECT 

FROM humanresources.employee AS employee

						 
-- Write a query to list all employees along with their associated sales orders. Include employees who may not have any sales orders. 
-- Use the COALESCE function to handle NULL values in the SalesOrderID column.

SELECT

FROM humanresources.employee AS employee

ORDER BY employee.employeeid;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- JOINS: CROSS JOINS

-- Explanation: A CROSS JOIN in SQL combines every row from the first table with every row from the second table. This type of join creates a Cartesian product, 
-- meaning that if the first table has 10 rows and the second table has 5 rows, the result will have 10 * 5 = 50 rows. 
-- A CROSS JOIN does not require any relationship or matching columns between the two tables.

-- Example: Good for arranging one person to meet multiple people

-- Write a query to generate all possible combinations of product categories and product models. Show the category name and the model name.

SELECT*
FROM production.productcategory AS productcategory

SELECT*
FROM production.productmodel AS productionmodel
LIMIT 10;

SELECT
	productcategory.name AS categoryname,
	productionmodel.name as modelname
FROM production.productcategory AS productcategory
CROSS JOIN production.productmodel AS productionmodel
ORDER BY productcategory.name ASC;
-- Each category name is matched to each model name

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- UNION, stacking the tables on top of each other without having duplicates
SELECT
	firstname,
	lastname,
	CONCAT(firstname, ' ', lastname) AS fullname
FROM person.person
UNION 
SELECT
	firstname,
	lastname,
	CONCAT(firstname, ' ', lastname) AS fullname
FROM person.person;


-- Union them together segregating employee and customer

SELECT *
FROM person.person;

SELECT*
FROM humanresources.employee

SELECT *
FROM sales.customer;

SELECT
	firstname,
	lastname,
	CONCAT(firstname, ' ', middlename, ' ', lastname)AS fullname,
	'Employee' AS category
FROM person.person AS person
INNER JOIN humanresources.employee as employee
	ON person.businessentityid = employee.businessentityid

UNION 

SELECT
	firstname,
	lastname,
	CONCAT(firstname, ' ', middlename, ' ', lastname)AS fullname,
	'Customer' AS category
FROM person.person AS person
INNER JOIN sales.customer AS customer
	ON person.businessentityid = customer.personid
WHERE customer.storeid IS NULL;

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- UNION ALL: EVERYTHING

-- Write a query to retrieve all sales orders and purchase orders, displaying the order ID and order date. 
-- Use UNION ALL to combine the sales and purchase order data, keeping all duplicates.

SELECT 
	salesorderid AS orderid, 
	orderdate
FROM sales.salesorderheader

UNION ALL

SELECT 
	purchaseorderid AS orderid, 
	orderdate
FROM purchasing.purchaseorderheader;


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- STRING FUNCTION
-- DATE handling, CONCAT()

-- Getting parts of the date out

SELECT 
	EXTRACT(YEAR FROM orderdate) AS year,
	EXTRACT(QUARTER FROM orderdate) AS QUARTER,
	EXTRACT(MONTH FROM orderdate) AS MONTH,
	EXTRACT(WEEK FROM orderdate) AS WEEK,
	EXTRACT(DAY FROM orderdate) AS DAY,
	EXTRACT(HOUR FROM orderdate) AS HOUR,
	EXTRACT(MINUTE FROM orderdate) AS MINUTE,
	EXTRACT(SECOND FROM orderdate) AS SECOND,

	CAST(orderdate AS TIME)AS time,
	CAST(orderdate AS DATE)AS date
FROM sales.salesorderheader
LIMIT 10;

-- DATETIME manipulations

SELECT
	orderdate AT TIME ZONE 'UTC'AT TIME ZONE 'Asia/Singapore'as local_time,
	CURRENT_DATE AS today,
	CURRENT_DATE + INTERVAL '10days' AS add_days,
	CURRENT_DATE - INTERVAL '10days' AS minus_days,
	CURRENT_DATE + INTERVAL '1 month' AS add_month
FROM sales.salesorderheader
WHERE territoryid = 1
	AND EXTRACT(YEAR FROM orderdate) = 2011;

-- Use string functions to format employee names and email addresses
SELECT
	CAST(person.businessentityid AS int),
	CAST(person.businessentityid AS numeric)/2 AS numeric_id,
	CAST(person.businessentityid AS decimal)/2 AS decimal_id,
	CAST(person.businessentityid AS VARCHAR(100))AS varchar_id,

	UPPER(person.lastname)as upperlastname,
	LOWER(person.lastname)as lowerlastname,
	LENGTH(person.firstname)as firstnamelength,

	LEFT(emailaddress.emailaddress,10) as startemail,
	RIGHT(emailaddress.emailaddress,10) as startemail,
	SUBSTRING(emailaddress.emailaddress, 1, 5) AS partialemail,
	SUBSTRING(emailaddress.emailaddress, 3, 5) AS partialemail,
	REPLACE(emailaddress.emailaddress, '@adventure-works.com', '@gmail.com')AS new_email
FROM person.person AS person
INNER JOIN person.emailaddress AS emailaddress
	ON person.businessentityid = emailaddress.businessentityid;
LIMIT 10;

/*
CAST()
UPPER9
LOWER()
LENGTH
LEFT
RIGHT
SUBSTRING
REPLACE
CONCAT
*/

-- From the following table write a query in  SQL to find the  email addresses of employees and groups them by city. 
-- Return top ten rows.

SELECT 
	address.city, 

FROM person.businessentityaddress AS businessentityaddress
INNER JOIN person.address AS address


GROUP BY 
LIMIT 10;
-- STRING_
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

--CASE FUNCTION: CASE WHEN THEN ELSE END

-- Categorize products based on their list price
SELECT 
	productid,
	name,
	listprice,
	CASE
		WHEN listprice = 0 THEN 'Free'
		WHEN listprice<50 THEN 'Budget'
		WHEN listprice BETWEEN 50 AND 100 THEN 'Mid-Range'
		ELSE 'Premium'
	END AS price_category
FROM production.product
ORDER BY listprice DESC;

-- Write a query to categorize sales orders based on the total amount (TotalDue). If the total amount is less than 1000, categorize it as "Low", 
-- if it's between 1000 and 5000, categorize it as "Medium", and if it's greater than 5000, categorize it as "High".

SELECT 
    salesorderheader.salesorderid AS salesorderid, 
    salesorderheader.totaldue AS totaldue,

FROM sales.salesorderheader AS salesorderheader;

-- Q7: Write a query to calculate bonuses for each employee. The bonus is calculated based on both their total sales and their length of employment:

-- If an employee has sales greater than 500,000 and has been employed for more than 5 years, they get a 15% bonus.
-- If their sales are greater than 500,000 but they’ve been employed for less than 5 years, they get a 10% bonus.
-- If their sales are between 100,000 and 500,000, they get a 5% bonus, regardless of years of service.
-- If their sales are less than 100,000, they get no bonus.

-- A7:


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- If time permits:
-- Window Functions
-- AGGREGATE

-- Explanation:
/*
A window function in SQL allows you to perform calculations across a set of table rows that are somehow related to the current row. 
Unlike regular aggregate functions (such as SUM, COUNT, AVG), window functions do not group the result into a single output. 
Instead, they return a value for every row while using a "window" of rows to perform the calculation.
*/

-- Let’s say we want to calculate the running total of sales for each salesperson, partitioned by their ID (so each salesperson gets their own total), 
-- and ordered by the order date.



-- Retrieving distinct active employee names along with salary statistics per department:


	
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------