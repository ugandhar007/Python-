class Employee:
    count=0
    total_salary=0


    def __init__(self, name, family, salary, department):
        self.name =name
        self.family=family
        self.salary=salary
        self.department=department
        Employee.count+=1
        Employee.total_salary =Employee.total_salary+ self.salary

    def average(self):
        avg_salary=Employee.total_salary/Employee.count
        print("The Average Salary is: ",avg_salary)

class fulltime_employee(Employee):

    def __init__(self, name, family, salary, department):
        Employee.__init__(self, name, family, salary, department)

emp1=Employee("Dinesh", 4, 9000, "Tech")
emp2=fulltime_employee("Sand", 5, 2000, "Lab")
emp3=Employee("Ravi", 2, 3000, "Sup")

emp3.average()
