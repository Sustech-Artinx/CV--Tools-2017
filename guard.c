#include<sys/types.h>
#include<stdio.h>
#include<unistd.h>

//Execute ./test if it exit or dump.

int main()
{
	pid_t pid,pid1;
	for(;;){
		pid=fork();
		if(pid==0){	//child
			execl("/home/a343024559/c/test","./test",NULL);
		}
		else{	//parent
			wait(NULL);
		}
	}
	return 0;
}

