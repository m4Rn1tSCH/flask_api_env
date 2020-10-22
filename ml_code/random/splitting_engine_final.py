#If the income is very very low;test for a small amount
def one_trial(df,income,newincome,bills, bill_change_value):
    global m
    #import pandas as pd
    #import numpy as np
    
    emergency=0.10
    #Maintaining as a ratio as percentage comparison would not be possible in Python
    if newincome==0:
    #Bills with buffer
        nbill=(bills*income)+(0.10*bills*income)
        for index, row in df.iterrows():
            if income==row["Income"]:
                print("Income checked")
            
                if bills==row["Bills ratio"]:
                    print("Awesome, bills checked and is the same too!")
                    vault=df['Savings'].values[index]
                    cash=df['Pocket Money'].values[index]
                    m=vault
                    total=vault+emergency+cash+bills
                    if total<1:
                                    vault=vault+(1-(total))
                    return(vault, cash, emergency, bills)
                    
                    break 
                else:
                    if bills>df['Bills ratio'].values[index]:
                         if (income<1000 and bills>0.28) or (income>=1000 and bills>0.65):
                                    print("Your bills are too high for your income!Coming to your rescue now..")
                                    remb=income-(nbill-0.1*nbill)+((df['Savings'].values[index])*income)+(0.1*nbill)-(emergency*income)
                                    print(remb)
                                    cash=(0.8*remb)/income   
                                    vault=(0.2*remb)/income
                                    m=vault
                                    total=vault+emergency+cash+bills
                                    if total<1:
                                        vault=vault+(1-(total))
                                    return(vault, cash, emergency, bills)
                                    break
                                    #df1 =pd.read_csv(r'Newbill.csv',delimiter=',')
                                    #df1=df1.append({'Income':income,'Bills ratio':bills,'Pocket Money':cash,'Savings':vault},ignore_index=True,sort=True)
                                    #df=df.append(df1)
                         else:
                             #bills=df['Bills ratio'].values[index]
                             rem=income-(nbill)-(emergency*income)
                             if rem>0:
                                            vault=(0.2*rem)/income
                                            cash=(0.8*rem)/income
                                            m=vault
                                            total=vault+emergency+cash+bills
                                            if total<1:
                                                vault=vault+(1-(total))
                                            return(vault, cash, emergency, bills)
                                            print(vault)
                                            print(cash)
                             else:
                                            #Rethink taking emergency=0
                                            remb=income-(nbill-0.1*nbill)+(0.1*nbill)+(emergency*income)
                                            emergency=0
                                            vault=(0.2*remb)/income
                                            cash=(0.8*remb)/income
                                            total=vault+emergency+cash+bills
                                            if total<1:
                                                vault=vault+(1-(total))
                                            print(vault)
                                            print(cash)
                                            m=vault
                                            return(vault, cash, emergency, bills)
            else:
                        #for index, row in df.iterrows():
                            #print("Checking income difference")
                            df['Income difference']=income-df['Income']
                            k=df[df['Income difference']>0].idxmin()
                            t=k['Income difference']
                            print(t)
                            if bills>df['Bills ratio'].values[t]:
                                if (income<1000 and bills>0.28) or (income>=1000 and bills>0.65):
                                    print("Your bills are too high for your income!Coming to your rescue now..")
                                    remb=income-(nbill-0.1*nbill)+((df['Savings'].values[t])*income)+(0.1*nbill)-(emergency*income)
                                    print(remb)
                                    cash=(0.8*remb)/income   
                                    vault=(0.2*remb)/income
                                    m=vault
                                    with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                    return(vault, cash, emergency, bills)
                                    break
                                    #df1 =pd.read_csv(r'Newbill.csv',delimiter=',')
                                    #df1=df1.append({'Income':income,'Bills ratio':bills,'Pocket Money':cash,'Savings':vault},ignore_index=True,sort=True)
                                    #df=df.append(df1)
                                else:
                             #bills=df['Bills ratio'].values[index]
                                 rem=income-(nbill)-(emergency*income)
                                 if rem>0:
                                            vault=(0.2*rem)/income
                                            cash=(0.8*rem)/income
                                            print(vault)
                                            print(cash)
                                            m=vault
                                            with open('Proportions-Auto recovered.csv','a') as newFile:
                                                newFileWriter=csv.writer(newFile)
                                                newFileWriter.writerow([income,vault,emergency,cash,bills])
                                            return(vault, cash, emergency, bills)
                            else:
                             #bills=df['Bills ratio'].values[index]
        
                                vault=df['Savings'].values[t]
                                cash=df['Pocket Money'].values[t]
                                total=vault+emergency+cash+bills
                                if total<1:
                                    vault=vault+(1-(total))
                                print(vault)
                                print(cash)
                                m=vault
                                with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                return(vault, cash, emergency, bills)
                                break
    
    #If there is varying income
    elif newincome!=0:
         
         print("Bills=",bills)  
         #newincome=input("What is your income this month?")
         #newincome=float(newincome)
         #bill_change=input("Is there a bill change from your previous month?")
         #Checking if the newincome is also already in the DB;if thats the case simple allocation
         if income!=newincome:
             for index, row in df.iterrows():
                 if newincome==row["Income"]:
                     if bills==row["Bills ratio"]:
                         vault=df['Savings'].values[index]
                         cash=df['Pocket Money'].values[index]
                         m=vault
                         total=vault+emergency+cash+bills
                         if total<1:
                                    vault=vault+(1-(total))
                         return(vault, cash, emergency, bills)
                         #bills=df['Bills ratio'].values[index]
                     else:
                         if bill_change_value !=0:
                             bill_change_value=float(bill_change_value)
                             bill_change_value=(float(bill_change_value)/float(newincome))
                             print("Checking bill change with varying income")
                             #bill_change_value=input("What do your new bills this month come to?")
                             #bill_change_value=float(bill_change_value)
                             #bill_change_value=round((float(bill_change_value)/float(newincome)),2)
                             nbill_change=(bill_change_value*newincome)+(0.10*bill_change_value*newincome)
                             if newincome<income and bill_change_value>=bills:
                              if (newincome<1000 and bill_change_value>0.28) or (newincome>=1000 and bill_change_value>0.65):
                                    print("Your bills are too high for your income!Coming to your rescue now1..")
                                    if (vault*income)>=700:
                                         print(vault)
                                         remb=newincome-(nbill_change-0.1*nbill_change)+400+(0.1*nbill_change)-(emergency*income)
                                         cash=(0.8*remb)/newincome
                                         vault=(0.2*remb)/newincome
                                         m=vault
                                         return(vault, cash, emergency, bills)
                           
                                    else:
                                        print(vault)
                                        remb=newincome-(nbill_change-0.1*nbill_change)+(vault*income)+(0.1*nbill_change)-(emergency*income)
                                        cash=(0.8*remb)/newincome
                                        vault=(0.2*remb)/newincome
                                        m=vault
                                        return(vault, cash, emergency, bills)
                              else:
                                rem=newincome-(bill_change_value*newincome)-(emergency*newincome)
                                if rem>0:
                                    vault=0.2*rem/newincome
                                    cash=0.8*rem/newincome
                                    m=vault
                                    return(vault, cash, emergency, bills)
                                else:
                                    #Need to rethink contingency=0
                                    remb=newincome-(nbill_change-0.1*nbill_change)+(0.1*nbill_change)+(emergency*income)
                                    emergency=0
                                    vault=0.2*remb/newincome
                                    cash=0.8*remb/newincome
                                    m=vault
                                    return(vault, cash, emergency, bills)
                                    
                         else:
                             if bill_change_value==0:
                                 if newincome>income:
                                     print("That's awesome!,More of guilt free spending!")
                                     rem=newincome-(bills*newincome)-(emergency*newincome)
                                     if rem>0:
                                         vault=0.2*rem/newincome
                                         cash=0.8*rem/newincome
                                         m=vault
                                         return(vault, cash, emergency, bills)
                                     else:
                                    #Need to rethink contingency=0
                                        nbills=(bills*newincome)+(0.10*bills*newincome)
                                        remb=newincome-(nbills-0.1*nbills)+(0.1*nbills)+(emergency*income)
                                        emergency=0
                                        vault=0.2*remb/newincome
                                        cash=0.8*remb/newincome
                                        m=vault
                                        return(vault, cash, emergency, bills)
                                 else:
                                     rem=newincome-(bills*newincome)-(emergency*newincome)
                                     if rem>0:
                                         vault=0.2*rem/newincome
                                         cash=0.8*rem/newincome
                                         m=vault
                                         return(vault, cash, emergency, bills)
                                     else:
                                    #Need to rethink contingency=0
                                        remb=income-(nbills-0.1*nbills)+(0.1*nbills)+(emergency*income)
                                        emergency=0
                                        vault=0.2*remb/income
                                        cash=0.8*remb/income
                                        m=vault
                                        return(vault, cash, emergency, bills)
                                    
                                    
                        #else:
                             #bills=df['Bills ratio'].values[index]
                 else:
                     #print("Checking income difference")
                     df['Income difference']=income-df['Income']
                     k=df[df['Income difference']>0].idxmin()
                     t=k['Income difference']
                     #vault=df['Savings'].values[t]
                     #print(t)
                     if bill_change_value !=0:
                             bill_change_value=float(bill_change_value)
                             bill_change_value=(float(bill_change_value)/float(newincome))
                             print("Checking bill change with varying income1")
                             #bill_change_value=input("What do your new bills this month come to?")
                             #bill_change_value=float(bill_change_value)
                             #bill_change_value=round((float(bill_change_value)/float(newincome)),2)
                             nbill_change=(bill_change_value*newincome)+(0.10*bill_change_value*newincome)
                             if newincome<income and bill_change_value>=bills:
                                if bill_change_value>df['Bills ratio'].values[t]:
                                    if (newincome<1000 and bill_change_value>0.28) or (newincome>=1000 and bill_change_value>0.65):
                                        print("Your bills are too high for your income!Coming to your rescue now2..")
                                        if (m*income)>=700:
                                            print(m)
                                            remb=newincome-(nbill_change-0.1*nbill_change)+400+(0.1*nbill_change)-(emergency*newincome)
                                            cash=(0.8*remb)/newincome
                                            vault=(0.2*remb)/newincome
                                            m=vault
                                            with open('Proportions-Auto recovered.csv','a') as newFile:
                                                 newFileWriter=csv.writer(newFile)
                                                 newFileWriter.writerow([income,vault,emergency,cash,bills])
                                            return(vault, cash, emergency, bill_change_value)
                                            break
                                        else:
                                            print(m)
                                            remb=newincome-(nbill_change-0.1*nbill_change)+(m*newincome)+(0.1*nbill_change)-(emergency*newincome)
                                            cash=(0.8*remb)/newincome
                                            vault=(0.2*remb)/newincome
                                            m=vault
                                            with open('Proportions-Auto recovered.csv','a') as newFile:
                                                 newFileWriter=csv.writer(newFile)
                                                 newFileWriter.writerow([income,vault,emergency,cash,bills])
                                            return(vault, cash, emergency, bill_change_value)
                                            break
                                    else:
                                        rem=newincome-(bill_change_value*newincome)-(emergency*newincome)
                                        if rem>0:
                                            vault=0.2*rem/newincome
                                            cash=0.8*rem/newincome
                                            m=vault
                                            with open('Proportions-Auto recovered.csv','a') as newFile:
                                                 newFileWriter=csv.writer(newFile)
                                                 newFileWriter.writerow([income,vault,emergency,cash,bills])
                                            return(vault, cash, emergency, bill_change_value)
                                            break
                                        else:
                                    #Need to rethink contingency=0
                                            remb=newincome-(nbill_change-0.1*nbill_change)+(0.1*nbill_change)+(emergency*newincome)
                                            emergency=0
                                            vault=0.2*remb/newincome
                                            cash=0.8*remb/newincome
                                            m=vault
                                            with open('Proportions-Auto recovered.csv','a') as newFile:
                                                 newFileWriter=csv.writer(newFile)
                                                 newFileWriter.writerow([income,vault,emergency,cash,bills])
                                            return(vault, cash, emergency, bill_change_value)
                                            break
                                    #break    
                             else:
                                rem=newincome-(bill_change_value*newincome)-(emergency*newincome)
                                if rem>0:
                                    vault=0.2*rem/newincome
                                    cash=0.8*rem/newincome
                                    m=vault
                                    with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                    return(vault, cash, emergency, bill_change_value)
                                    break
                                else:
                                    #Need to rethink contingency=0
                                    remb=income-(nbill_change-0.1*nbill_change)+(0.1*nbill_change)+(emergency*income)
                                    emergency=0
                                    vault=0.2*remb/income
                                    cash=0.8*remb/income
                                    m=vault
                                    with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                    return(vault, cash, emergency, bill_change_value)
                                    break
                     else:
                             if bill_change_value==0:
                                 print("Yayy")
                                 if newincome>income:
                                     print("That's awesome!,More of guilt free spending!")
                                     rem=newincome-(bills*newincome)-(emergency*newincome)
                                     if rem>0:
                                         vault=0.2*rem/newincome
                                         cash=0.8*rem/newincome
                                         m=vault
                                         with open('Proportions-Auto recovered.csv','a') as newFile:
                                              newFileWriter=csv.writer(newFile)
                                              newFileWriter.writerow([income,vault,emergency,cash,bills])
                                         return(vault, cash, emergency, bills)
                                         break
                                     else:
                                    #Need to rethink contingency=0
                                        nbills=(bills*newincome)+(0.10*bills*newincome)
                                        remb=newincome-(nbills-0.1*nbills)+(0.1*nbills)+(emergency*newincome)
                                        emergency=0
                                        vault=0.2*remb/newincome
                                        cash=0.8*remb/newincome
                                        m=vault
                                        with open('Proportions-Auto recovered.csv','a') as newFile:
                                            newFileWriter=csv.writer(newFile)
                                            newFileWriter.writerow([income,vault,emergency,cash,bills])
                                        return(vault, cash, emergency, bills)
                                        break
                                 else:
                                     rem=newincome-(bills*newincome)-(emergency*newincome)
                                     if rem>0:
                                         vault=0.2*rem/newincome
                                         cash=0.8*rem/newincome
                                         m=vault
                                         with open('Proportions-Auto recovered.csv','a') as newFile:
                                              newFileWriter=csv.writer(newFile)
                                              newFileWriter.writerow([income,vault,emergency,cash,bills])
                                         return(vault, cash, emergency, bills)
                                         break
                                     else:
                                    #Need to rethink contingency=0
                                        remb=newincome-(nbills-0.1*nbills)+(0.1*nbills)+(emergency*newincome)
                                        emergency=0
                                        vault=0.2*remb/newincome
                                        cash=0.8*remb/newincome
                                        m=vault
                                        with open('Proportions-Auto recovered.csv','a') as newFile:
                                             newFileWriter=csv.writer(newFile)
                                             newFileWriter.writerow([income,vault,emergency,cash,bills])
                                        return(vault, cash, emergency, bills)
                                        break
                    
         else:
             print("Your income is the same as last month! You are in a good place!:)")
             for index, row in df.iterrows():
                 if income==row["Income"]:
                     print("Income checked")
                     if bills==row["Bills ratio"]:
                        print("Awesome, thanks for the info!")
                        vault=df['Savings'].values[index]
                        cash=df['Pocket Money'].values[index]
                        m=vault
                        return(vault, cash, emergency, bills)
                        break 
                     else:
                        vault=df['Savings'].values[index]
                        print(vault)
                        vault=vault+(bills-(df['Bills ratio'].values[index]))
                        m=vault
                        print(vault)
                        cash=df['Pocket Money'].values[index]
                        with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                        m=vault
                        return(vault, cash, emergency, bills)
                        break
                 else:
            
                     df['Income difference']=income-df['Income']
                     t=df[df['Income difference'] > 0].idxmin()
                     print(t)
                     if income>df['Income'].values[t]:
                         if bills>df['Bills ratio'].values[t]:
                             if (income<1500 and bills>0.28) or income>1500 and bills>0.65:
                                 print("Your bills are too high for your income!We will help you!")
                                 
                                 if (vault*income)>=700:
                                         print(vault)
                                         remb=newincome-(nbill_change-0.1*nbill_change)+400+(0.1*nbill_change)-(emergency*income)
                                         cash=(0.8*remb)/newincome
                                         vault=(0.2*remb)/newincome
                                         print(cash)
                                         m=vault
                                         with open('Proportions-Auto recovered.csv','a') as newFile:
                                              newFileWriter=csv.writer(newFile)
                                              newFileWriter.writerow([income,vault,emergency,cash,bills])
                                         return(vault, cash, emergency, bills)
                                         break
                           
                                 else:
                                        print(vault)
                                        remb=newincome-(nbill_change-0.1*nbill_change)+(vault*income)+(0.1*nbill_change)-(emergency*income)
                                        cash=(0.8*remb)/newincome
                                        vault=(0.2*remb)/newincome
                                        m=vault
                                        with open('Proportions-Auto recovered.csv','a') as newFile:
                                             newFileWriter=csv.writer(newFile)
                                             newFileWriter.writerow([income,vault,emergency,cash,bills])
                                        return(vault, cash, emergency, bills)
                                        print(cash)
                                        break
                                 
                             else:
                                 bills=df['Bills ratio'].values[t]
                                 rem=income-(bills*income)-(emergency*income)
                                 if rem>0:
                                     vault=0.2*rem/income
                                     cash=0.8*rem/income
                                     m=vault
                                     with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                     return(vault, cash, emergency, bills)
                                     break
                                 else:
                                     remb=income-(nbill-0.1*nbill)+(0.1*nbill)+(emergency*income)
                                     emergency=0
                                     vault=0.2*remb/income
                                     cash=0.8*remb/income
                                     m=vault
                                     with open('Proportions-Auto recovered.csv','a') as newFile:
                                        newFileWriter=csv.writer(newFile)
                                        newFileWriter.writerow([income,vault,emergency,cash,bills])
                                     return(vault, cash, emergency, bills)
                                     break
    #df1 =pd.read_csv(r'Newbill.csv',delimiter=',')
    #df1=df1.append({'Income':income,'Bills ratio':bills,'Contingency':emergency,'Pocket Money':cash,'Savings':vault},ignore_index=True,sort=True)
    #df=df.append(df1) 
import pandas as pd
import csv
df=pd.read_csv(r'Proportions-Auto recovered.csv',delimiter=',')
    #print(df)
income=input("What's your income?")
income=int(income,10)
bills=input("What do your bills come to?")      
bills=(float(bills)/float(income))
m=0
#total=0
#newincome=0
#bill_change_value=0
t1=one_trial(df,income,newincome,bills, bill_change_value)