#ifndef SIGINT_HANDLER_H
#define SIGINT_HANDLER_H

#include <signal.h>

static volatile sig_atomic_t interrupted;
static void (*old_handler)(int);

static void signal_handler(int signo) {
  interrupted = 1;
}

static void sigint_on(void) {
  interrupted = 0;
  old_handler = signal(SIGINT, signal_handler);
}

static void sigint_off(void) {
  if(old_handler != SIG_ERR) {
    signal(SIGINT, old_handler);
    if(interrupted) raise(SIGINT);
  }
}

#endif
