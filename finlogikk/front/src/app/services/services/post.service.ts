
import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable, tap } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PostService {
  private API_URL = "http://localhost:8088/api/v1/post";
  private API_URL1 = "http://localhost:8088/api/v1/votePost";
  constructor(private httpClient: HttpClient) { }

  getAllPosts(){
    return this.httpClient.get(`${this.API_URL}/getAllPosts`)
  }
  addPost(post : any, IdUser : any , IdTopic : number) {
    return this.httpClient.post(`${this.API_URL}/addPost/${IdUser}/${IdTopic}`, post)
  }
  updatePost(post : any, IdPost : number){
    return  this.httpClient.put(`${this.API_URL}/updatePost/${IdPost}`, post)
  }
  deletePost(IdPost : number){
    return  this.httpClient.delete(`${this.API_URL}/deletePost/${IdPost}`)
  }
  getPost(IdPost : any){
    return  this.httpClient.get(`${this.API_URL}/getPost/${IdPost}`)
  }

  getPostsByTopicId(IdTopic : any){
    return this.httpClient.get(`${this.API_URL}/byTopic/${IdTopic}`)
  }
  commentCount(IdPost : any){
    return  this.httpClient.get(`${this.API_URL}/commentCount/${IdPost}`)
  }

  voteLike(IdPost:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteLike/${IdPost}/${IdUser}`,null);
  }
  voteDislike(IdPost:any,IdUser:any){
    return this.httpClient.post(`${this.API_URL1}/voteDislike/${IdPost}/${IdUser}`,null);
  }
  getUserVoteStatus1(PostId: any, userId: any): Observable<string> {
    const url = `${this.API_URL1}/status/${PostId}/${userId}`;
    return this.httpClient.get(url, { responseType: 'text' });
  }



}
