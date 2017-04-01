/*Name: A C++ program that could find if two vectors are same.
File name: 1507243_1-1.cpp 
Created by Li Junyan, ID number: 1507243.
Description: This program asks user to randomly input numbers of two vectors, and then compares
			 all elements in both vectors and tell if they are the same.
*/

#include <vector>
#include <iostream>
using namespace std;//use the namespace of std

bool same_vec(vector<int> a, vector<int> b)
{
	int n, m;
	//The following code block is used to compare if there are any similar elements in both vectors.
	for (n = 0; n < a.size(); n++)//a for loop to traversal all elements in vector a
	{     
		for (m = 0; m < b.size(); m++)//a for loop to traversal all elements in vector b
		{
			if (a[n] == b[m])//judge if the nth element in a equals to the mth one in b
				break;
		}
		if (m == b.size())//judge if m equals the size of b
			break;
	}
	if (n < a.size())
		return 0;//boolean value is false if any difference found
	else if (n == a.size())
		return 1;//boolean value is true if none difference found
}

int main()
{
	vector<int> a, b;//define vectors a and b 
	int n;
	cout << "Please enter numbers in a line£¨space between each number£©: "<<endl;
	do{
		cin >> n;
		a.push_back(n);//acquire a elements from keyboard
	} while (getchar() != '\n');//store the value a
	cout << "Please enter numbers in a line£¨space between each number£©: "<<endl;
	do{
		cin >> n;
		b.push_back(n);//acquire b elements from keyboard    
	} while (getchar() != '\n');//store the value b
	if (same_vec(a, b))//compare whether a and b are the same
		cout << "vector A and B are same." << endl;
	else
		cout << "vector A and B are not same." << endl;
}
