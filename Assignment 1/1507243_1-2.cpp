/*Name: A C++ program that could find if two vectors are same.
File name: 1507243_1-2.cpp
Created by Li Junyan, ID number: 1507243.
Description: This program asks user to randomly input numbers of two vectors, and then compares
			 all elements in both vectors and tell if they are the same.
*/

#include <iostream>
#include<string.h>
using namespace std;

char *findC(char *const source, char *const obj)
//The following code block is used to search the desired character.
{ 
	int i, j;
	for (i = 0; i < strlen(source); i++)
	{
		for (j = 0; j < strlen(obj); j++)//2 for loops to find the 
		{
			if (source[i] == obj[j])
			{
				return &source[i];//return the address if any matchiong character found 
			}
			else
			{
				break;
			}
		}		
	}
	return NULL;
}

int main(void)
{
	char source[100];
	char obj[100];
	char *pt;
	cout << "Please input the initial string : " << endl; //ask the usert to enter the string
	cin >> source;
	cout << "Please input the desired string : " << endl; //ask the usert to enter the designed character
	cin >> obj;
	pt = findC(source, obj);
	if (pt != NULL)
		cout << "The address of the desired string is : " << (void *)pt << endl; //dispaly the reslut
	else
		cout << "The string cannot be found!" << endl;//show error
}
