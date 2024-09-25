import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class TopicService {
  private API_URL = "http://localhost:8088/api/v1/topic";
  private API_URL1 = "http://localhost:8088/api/v1/voteTopic";
  constructor(private httpClient: HttpClient) { }

  getAllTopics(){
    return this.httpClient.get(`${this.API_URL}/getAllTopics`)
  }
  addTopic(topic : any, IdUser : any) {
    return this.httpClient.post(`${this.API_URL}/addTopic/${IdUser}`, topic)
  }
  deleteTopic(IdTopic : number){
    return  this.httpClient.delete(`${this.API_URL}/deleteTopic/${IdTopic}`)
  }
  getTopic(IdTopic : any){
    return  this.httpClient.get(`${this.API_URL}/getTopic/${IdTopic}`)
  }

  postCount(IdTopic : any){
    return  this.httpClient.get(`${this.API_URL}/postCount/${IdTopic}`)
  }

  voteLike(IdTopic:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteLike/${IdTopic}/${IdUser}`,null);
  }
  voteDislike(IdTopic:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteDislike/${IdTopic}/${IdUser}`,null);
  }
  getUserVoteStatus(topicId: any, userId: any): Observable<string> {
    const url = `${this.API_URL1}/status/${topicId}/${userId}`;
    return this.httpClient.get(url, { responseType: 'text' });
  }


}
