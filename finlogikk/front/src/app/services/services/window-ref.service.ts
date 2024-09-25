import { Injectable } from '@angular/core';

function getWindow(): any {
  return window;
}

@Injectable({
  providedIn: 'root',
})
export class WindowRef {
  get nativeWindow(): any {
    return getWindow();
  }
}
