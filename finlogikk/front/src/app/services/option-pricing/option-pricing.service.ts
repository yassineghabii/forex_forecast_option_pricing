import { Injectable } from '@angular/core';
import {HttpClient, HttpHeaders} from "@angular/common/http";
import {Observable} from "rxjs";

@Injectable({
  providedIn: 'root'
})
export class OptionPricingService {

  private apiUrl = 'http://localhost:5000/calculate';

  constructor(private http: HttpClient) { }

  calculateOption(data: any): Observable<any> {
    const headers = new HttpHeaders().set('Content-Type', 'application/json');
    return this.http.post(this.apiUrl, data, { headers });
  }

}
